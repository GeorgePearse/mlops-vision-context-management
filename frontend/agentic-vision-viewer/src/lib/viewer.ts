import type { ViewerArtifact, ViewerEvent } from '$lib/types';

export const STREAM_EVENT_TYPES = [
  'tool_called',
  'tool_result',
  'overlay_generated',
  'log',
  'frame_started',
  'frame_completed',
  'run_completed',
  'run_failed',
  'run_terminal',
] as const;

export function formatTimestamp(
  value: string,
  options: Intl.DateTimeFormatOptions = {},
): string {
  return new Date(value).toLocaleString([], options);
}

export function appendDedupedEvents(
  current: ViewerEvent[],
  incoming: ViewerEvent[],
): ViewerEvent[] {
  const merged = new Map<number, ViewerEvent>();

  for (const event of current) {
    merged.set(event.sequence, event);
  }

  for (const event of incoming) {
    merged.set(event.sequence, event);
  }

  return Array.from(merged.values()).toSorted(
    (left, right) => left.sequence - right.sequence,
  );
}

export function resolveArtifactFromEvent(
  event: ViewerEvent | null,
): ViewerArtifact | null {
  return event?.artifact || null;
}

export function summarizePayload(
  value: Record<string, unknown> | null | undefined,
): string {
  if (!value) {
    return 'No structured payload';
  }

  return JSON.stringify(value, null, 2);
}
