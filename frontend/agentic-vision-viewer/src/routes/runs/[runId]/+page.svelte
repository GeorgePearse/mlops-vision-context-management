<script lang="ts">
  import { resolve } from '$app/paths';
  import { onDestroy, onMount } from 'svelte';

  import {
    buildArtifactUrl,
    buildSseUrl,
    fetchRun,
    fetchRunEvents,
  } from '$lib/api';
  import type { ViewerArtifact, ViewerEvent, ViewerRun } from '$lib/types';
  import {
    STREAM_EVENT_TYPES,
    appendDedupedEvents,
    formatTimestamp,
    resolveArtifactFromEvent,
    summarizePayload,
  } from '$lib/viewer';

  import type { PageData } from './$types';

  export let data: PageData;

  let run: ViewerRun | null = null;
  let events: ViewerEvent[] = [];
  let selectedEventSequence: number | null = null;
  let selectedEvent: ViewerEvent | null = null;
  let overlayEvents: ViewerEvent[] = [];
  let selectedArtifact: ViewerArtifact | null = null;
  let error: string | null = null;

  let closeStream = () => {};

  $: selectedEvent =
    events.find((event) => event.sequence === selectedEventSequence) || null;
  $: overlayEvents = events.filter(
    (event) => event.event_type === 'overlay_generated',
  );
  $: selectedArtifact =
    resolveArtifactFromEvent(selectedEvent) ||
    resolveArtifactFromEvent(overlayEvents.at(-1) || null);

  onMount(() => {
    void loadInitialState();

    return () => {
      closeStream();
    };
  });

  onDestroy(() => {
    closeStream();
  });

  async function refreshRun(): Promise<void> {
    try {
      run = await fetchRun(data.runId);
    } catch {
      // Ignore refresh failures while an SSE session is active.
    }
  }

  function openStream(afterSequence: number): void {
    closeStream();

    const eventSource = new EventSource(buildSseUrl(data.runId, afterSequence));

    const handleStreamEvent = (messageEvent: MessageEvent<string>) => {
      const parsed = JSON.parse(messageEvent.data) as
        | ViewerEvent
        | { status?: string };

      if ('sequence' in parsed) {
        events = appendDedupedEvents(events, [parsed]);

        if (parsed.event_type === 'overlay_generated') {
          selectedEventSequence = parsed.sequence;
        }

        if (
          parsed.event_type === 'run_completed' ||
          parsed.event_type === 'run_failed'
        ) {
          void refreshRun();
        }

        return;
      }

      if (parsed.status && run) {
        run = { ...run, status: parsed.status };
      }
    };

    const handleStreamError = () => {
      eventSource.close();
    };

    for (const eventType of STREAM_EVENT_TYPES) {
      eventSource.addEventListener(eventType, handleStreamEvent as EventListener);
    }

    eventSource.addEventListener('error', handleStreamError);

    closeStream = () => {
      for (const eventType of STREAM_EVENT_TYPES) {
        eventSource.removeEventListener(
          eventType,
          handleStreamEvent as EventListener,
        );
      }

      eventSource.removeEventListener('error', handleStreamError);
      eventSource.close();
      closeStream = () => {};
    };
  }

  async function loadInitialState(): Promise<void> {
    error = null;

    try {
      const [nextRun, nextEvents] = await Promise.all([
        fetchRun(data.runId),
        fetchRunEvents(data.runId),
      ]);

      run = nextRun;
      events = nextEvents;

      const lastSequence =
        nextEvents.length > 0 ? nextEvents[nextEvents.length - 1].sequence : 0;
      const latestOverlay = nextEvents
        .toReversed()
        .find((event) => event.event_type === 'overlay_generated');

      if (latestOverlay) {
        selectedEventSequence = latestOverlay.sequence;
      } else if (nextEvents.length > 0) {
        selectedEventSequence = nextEvents[nextEvents.length - 1].sequence;
      }

      openStream(lastSequence);
    } catch (nextError) {
      error =
        nextError instanceof Error ? nextError.message : 'Failed to load run';
    }
  }
</script>

<svelte:head>
  <title>
    {run?.run_label || run?.frame_id || data.runId} · Agentic Vision Viewer
  </title>
</svelte:head>

<main class="viewer-shell run-detail-shell">
  <section class="detail-header panel-card">
    <div>
      <a class="back-link" href={resolve('/')}>Back to runs</a>
      <p class="panel-kicker">Run</p>
      <h1>{run?.run_label || run?.frame_id || data.runId}</h1>
      <p class="viewer-subtitle">{run?.frame_uri || 'Loading frame URI…'}</p>
    </div>
    <div class="detail-header-meta">
      <span class={`status-pill status-${run?.status || 'queued'}`}>
        {run?.status || 'queued'}
      </span>
      <span>
        {run
          ? formatTimestamp(run.updated_at, {
              hour: 'numeric',
              minute: '2-digit',
              second: '2-digit',
            })
          : '…'}
      </span>
    </div>
  </section>

  {#if error}
    <p class="error-banner">{error}</p>
  {/if}

  <section class="detail-grid">
    <aside class="panel-card artifact-rail">
      <div class="panel-header">
        <div>
          <p class="panel-kicker">Artifacts</p>
          <h2>Stage overlays</h2>
        </div>
      </div>

      <div class="artifact-list">
        {#if overlayEvents.length === 0}
          <p class="empty-state">No overlay artifacts yet.</p>
        {/if}

        {#each overlayEvents as event (event.sequence)}
          {#if event.artifact}
            <button
              class={`artifact-card ${selectedEventSequence === event.sequence ? 'artifact-card-active' : ''}`}
              type="button"
              on:click={() => {
                selectedEventSequence = event.sequence;
              }}
            >
              <span class="artifact-stage">
                {event.stage_name || event.artifact.artifact_kind}
              </span>
              <strong>{event.artifact.artifact_kind}</strong>
              <span class="artifact-meta">
                {event.artifact.width}×{event.artifact.height}
              </span>
            </button>
          {/if}
        {/each}
      </div>
    </aside>

    <section class="panel-card main-viewport">
      <div class="panel-header">
        <div>
          <p class="panel-kicker">Viewport</p>
          <h2>{selectedEvent?.stage_name || 'Latest overlay'}</h2>
        </div>
      </div>

      <div class="image-stage">
        {#if selectedArtifact}
          <img
            alt={selectedArtifact.artifact_kind}
            class="artifact-image"
            src={buildArtifactUrl(data.runId, selectedArtifact.artifact_id)}
          />
        {:else}
          <div class="empty-state image-empty">
            Waiting for the first rendered artifact…
          </div>
        {/if}
      </div>

      <div class="detail-panels">
        <div class="detail-panel">
          <p class="panel-kicker">Selected event</p>
          <pre>
{selectedEvent ? summarizePayload(selectedEvent.payload ?? null) : 'No event selected.'}</pre
          >
        </div>
        <div class="detail-panel">
          <p class="panel-kicker">Final annotations</p>
          <pre>{run?.result_annotations || 'Run still in progress.'}</pre>
        </div>
      </div>
    </section>

    <aside class="panel-card event-stream">
      <div class="panel-header">
        <div>
          <p class="panel-kicker">Stream</p>
          <h2>Logs and tool calls</h2>
        </div>
      </div>

      <div class="event-list">
        {#if events.length === 0}
          <p class="empty-state">Waiting for events…</p>
        {/if}

        {#each events as event (event.sequence)}
          <button
            class={`event-card ${selectedEventSequence === event.sequence ? 'event-card-active' : ''}`}
            type="button"
            on:click={() => {
              selectedEventSequence = event.sequence;
            }}
          >
            <div class="event-card-top">
              <span class={`status-dot status-dot-${event.status || 'info'}`}></span>
              <strong>{event.stage_name || event.event_type}</strong>
              <span class="event-time">
                {formatTimestamp(event.timestamp, {
                  hour: 'numeric',
                  minute: '2-digit',
                  second: '2-digit',
                })}
              </span>
            </div>
            <p>{event.message || event.event_type}</p>
            <span class="event-type">{event.event_type}</span>
          </button>
        {/each}
      </div>
    </aside>
  </section>
</main>
