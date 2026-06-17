package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strconv"
	"sync"

	"github.com/google/uuid"
	"github.com/nextlevelbuilder/goclaw/internal/hooks"
	"github.com/nextlevelbuilder/goclaw/internal/providers"
	"github.com/nextlevelbuilder/goclaw/internal/store"
)

const maxSequentialWaitBatchMs = 300000

// ToolStage runs per iteration after PruneStage. Executes tool calls from
// ThinkState.LastResponse, checks exit conditions (loop kill, read-only streak, budget).
type ToolStage struct {
	deps   *PipelineDeps
	result StageResult
}

// NewToolStage creates a ToolStage.
func NewToolStage(deps *PipelineDeps) *ToolStage {
	return &ToolStage{deps: deps, result: Continue}
}

func (s *ToolStage) Name() string        { return "tool" }
func (s *ToolStage) Result() StageResult { return s.result }

// Execute extracts tool calls, dispatches them, checks exit conditions.
func (s *ToolStage) Execute(ctx context.Context, state *RunState) error {
	s.result = Continue

	resp := state.Think.LastResponse
	if resp == nil || len(resp.ToolCalls) == 0 {
		return nil // no tools — ThinkStage already set BreakLoop
	}

	toolCalls := resp.ToolCalls
	if s.deps.ExecuteToolCall == nil {
		return fmt.Errorf("ExecuteToolCall callback not configured")
	}

	// Parallel path: separate I/O (parallel) from state mutation (sequential).
	// Requires both ExecuteToolRaw and ProcessToolResult callbacks.
	if len(toolCalls) > 1 && s.deps.ExecuteToolRaw != nil && s.deps.ProcessToolResult != nil && !s.requiresSequential(toolCalls) {
		return s.executeParallel(ctx, state, toolCalls)
	}

	// Sequential fallback: ExecuteToolCall handles both I/O and state mutation.
	cumulativeWaitMs := 0
	for _, tc := range toolCalls {
		if s.shouldStopBeforeTool(ctx, state) {
			return nil
		}
		if s.deps.SequentialToolCall != nil && s.deps.SequentialToolCall(tc) {
			cumulativeWaitMs += toolCallTimeMs(tc)
			if cumulativeWaitMs > maxSequentialWaitBatchMs {
				s.result = AbortRun
				return nil
			}
		}

		// Hook: sync PreToolUse — block if hook denies. Builtin-source hooks may
		// rewrite tc.Arguments via UpdatedToolInput (e.g. path-sanitizer); apply
		// before ExecuteToolCall so the rewrite is authoritative.
		if r, _ := s.deps.FireHook(ctx, hooks.Event{
			EventID:   uuid.NewString(),
			SessionID: state.Input.SessionKey,
			TenantID:  store.TenantIDFromContext(ctx),
			AgentID:   store.AgentIDFromContext(ctx),
			ToolName:  tc.Name,
			ToolInput: tc.Arguments,
			HookEvent: hooks.EventPreToolUse,
		}); r.Decision == hooks.DecisionBlock {
			// Inject synthetic blocked tool message and skip actual execution.
			state.Messages.AppendPending(providers.Message{
				Role:       "tool",
				Content:    "Hook blocked: pre_tool_use",
				ToolCallID: tc.ID,
			})
			state.Tool.TotalToolCalls++
			continue
		} else if r.UpdatedToolInput != nil {
			tc.Arguments = r.UpdatedToolInput
		}

		msgs, err := s.deps.ExecuteToolCall(ctx, state, tc)
		if err != nil {
			// Tool execution failed. Emit an error tool result so the assistant's
			// tool_calls message stays paired — OpenAI-compatible providers reject a
			// message whose tool_call_ids lack tool responses (HTTP 400). Record the
			// failure and continue rather than aborting the whole run.
			slog.Warn("pipeline.tool.exec_failed",
				"tool", tc.Name, "tool_call_id", tc.ID, "error", err)
			state.Messages.AppendPending(toolErrorMessage(tc.ID, tc.Name, err))
			state.Tool.TotalToolCalls++
			continue
		}
		for _, msg := range msgs {
			state.Messages.AppendPending(msg)
		}
		state.Tool.TotalToolCalls++

		// Hook: async PostToolUse — fire and forget with detached context.
		if s.deps.Hooks != nil {
			detached := context.WithoutCancel(ctx)
			go s.deps.FireHook(detached, hooks.Event{ //nolint:errcheck
				EventID:   uuid.NewString(),
				SessionID: state.Input.SessionKey,
				TenantID:  store.TenantIDFromContext(ctx),
				AgentID:   store.AgentIDFromContext(ctx),
				ToolName:  tc.Name,
				ToolInput: tc.Arguments,
				HookEvent: hooks.EventPostToolUse,
			})
		}

		if state.Tool.LoopKilled {
			s.result = BreakLoop
			return nil
		}
		if ctx.Err() != nil {
			s.result = AbortRun
			return nil
		}
	}

	s.checkExitConditions(state)
	return nil
}

func (s *ToolStage) requiresSequential(toolCalls []providers.ToolCall) bool {
	if s.deps.SequentialToolCall == nil {
		return false
	}
	for _, tc := range toolCalls {
		if s.deps.SequentialToolCall(tc) {
			return true
		}
	}
	return false
}

func (s *ToolStage) shouldStopBeforeTool(ctx context.Context, state *RunState) bool {
	if ctx.Err() != nil {
		s.result = AbortRun
		return true
	}
	if s.deps.Config.MaxToolCalls > 0 && state.Tool.TotalToolCalls >= s.deps.Config.MaxToolCalls {
		s.result = BreakLoop
		return true
	}
	return false
}

func toolCallTimeMs(tc providers.ToolCall) int {
	v, ok := tc.Arguments["timeMs"]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case json.Number:
		i, err := strconv.Atoi(n.String())
		if err == nil {
			return i
		}
	}
	return 0
}

// toolErrorMessage builds a tool-result message for a tool_call whose execution
// failed, so every tool_call_id in an assistant message gets a paired response.
// Without it, OpenAI-compatible providers reject the next request with
// "tool_call_ids did not have response messages" (HTTP 400).
func toolErrorMessage(toolCallID, toolName string, err error) providers.Message {
	return providers.Message{
		Role:       "tool",
		ToolCallID: toolCallID,
		Content:    fmt.Sprintf("Tool %q failed to execute: %v", toolName, err),
	}
}

// executeParallel runs tool I/O concurrently, then processes results sequentially.
func (s *ToolStage) executeParallel(ctx context.Context, state *RunState, toolCalls []providers.ToolCall) error {
	type rawResult struct {
		tc      providers.ToolCall
		msg     providers.Message
		rawData any
		err     error
	}

	// Phase 1: parallel I/O (no state mutation)
	results := make([]rawResult, len(toolCalls))
	var wg sync.WaitGroup
	for i, tc := range toolCalls {
		wg.Add(1)
		go func(idx int, tc providers.ToolCall) {
			defer wg.Done()
			msg, rawData, err := s.deps.ExecuteToolRaw(ctx, tc)
			results[idx] = rawResult{tc: tc, msg: msg, rawData: rawData, err: err}
		}(i, tc)
	}
	wg.Wait()

	// Phase 2: sequential state mutation (safe, deterministic order)
	for _, r := range results {
		if r.err != nil {
			// Tool I/O failed. Still answer this tool_call_id with an error result so
			// the assistant's tool_calls message stays paired — an orphaned id makes
			// OpenAI-compatible providers 400 on the next request. Record and continue
			// so the rest of the batch is answered too (don't abort the whole run).
			slog.Warn("pipeline.tool.parallel_exec_failed",
				"tool", r.tc.Name, "tool_call_id", r.tc.ID, "error", r.err)
			state.Messages.AppendPending(toolErrorMessage(r.tc.ID, r.tc.Name, r.err))
			state.Tool.TotalToolCalls++
			continue
		}
		processed := s.deps.ProcessToolResult(ctx, state, r.tc, r.msg, r.rawData)
		for _, msg := range processed {
			state.Messages.AppendPending(msg)
		}
		state.Tool.TotalToolCalls++

		// Hook: async PostToolUse for parallel path — fire and forget.
		// PreToolUse is not instrumented in the parallel path (TODO: add when parallel path matures).
		if s.deps.Hooks != nil {
			detached := context.WithoutCancel(ctx)
			go s.deps.FireHook(detached, hooks.Event{ //nolint:errcheck
				EventID:   uuid.NewString(),
				SessionID: state.Input.SessionKey,
				TenantID:  store.TenantIDFromContext(ctx),
				AgentID:   store.AgentIDFromContext(ctx),
				ToolName:  r.tc.Name,
				ToolInput: r.tc.Arguments,
				HookEvent: hooks.EventPostToolUse,
			})
		}

		if state.Tool.LoopKilled {
			s.result = BreakLoop
			return nil
		}
	}

	s.checkExitConditions(state)
	return nil
}

// checkExitConditions checks read-only streak and tool budget.
func (s *ToolStage) checkExitConditions(state *RunState) {
	if state.Tool.LoopKilled {
		s.result = BreakLoop
		return
	}
	if s.deps.CheckReadOnly != nil {
		warningMsg, shouldBreak := s.deps.CheckReadOnly(state)
		if warningMsg != nil {
			state.Messages.AppendPending(*warningMsg)
		}
		if shouldBreak {
			s.result = BreakLoop
			return
		}
	}
	if s.deps.Config.MaxToolCalls > 0 && state.Tool.TotalToolCalls >= s.deps.Config.MaxToolCalls {
		s.result = BreakLoop
	}
}
