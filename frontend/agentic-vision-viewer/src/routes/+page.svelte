<script lang="ts">
  import { resolve } from '$app/paths';
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';

  import { fetchRuns, startRun } from '$lib/api';
  import type { StartRunPayload, ViewerRun } from '$lib/types';
  import { formatTimestamp } from '$lib/viewer';

  type LaunchFormState = {
    frame_uri: string;
    dataset_name: string;
    run_label: string;
    max_iters: number;
    sam3_handler_name: string;
  };

  const DEFAULT_FORM: LaunchFormState = {
    frame_uri: '',
    dataset_name: '',
    run_label: '',
    max_iters: 16,
    sam3_handler_name: 'premier',
  };

  let runs: ViewerRun[] = [];
  let loading = true;
  let submitting = false;
  let error: string | null = null;
  let formState: LaunchFormState = { ...DEFAULT_FORM };

  $: runCountLabel = `${runs.length} run${runs.length === 1 ? '' : 's'}`;

  onMount(() => {
    void loadRuns();
  });

  async function loadRuns(): Promise<void> {
    loading = true;
    error = null;

    try {
      runs = await fetchRuns();
    } catch (nextError) {
      error =
        nextError instanceof Error ? nextError.message : 'Failed to load runs';
    } finally {
      loading = false;
    }
  }

  async function handleSubmit(): Promise<void> {
    submitting = true;
    error = null;

    try {
      const payload: StartRunPayload = {
        frame_uri: formState.frame_uri,
        max_iters: formState.max_iters,
        sam3_handler_name: formState.sam3_handler_name,
        ...(formState.dataset_name
          ? { dataset_name: formState.dataset_name }
          : {}),
        ...(formState.run_label ? { run_label: formState.run_label } : {}),
      };

      const result = await startRun(payload);
      await goto(resolve(`/runs/${result.run_id}`));
    } catch (nextError) {
      error =
        nextError instanceof Error ? nextError.message : 'Failed to start run';
    } finally {
      submitting = false;
    }
  }
</script>

<svelte:head>
  <title>Agentic Vision Viewer</title>
</svelte:head>

<main class="viewer-shell">
  <section class="viewer-hero">
    <div class="viewer-hero-copy">
      <p class="viewer-eyebrow">Agentic Vision Viewer</p>
      <h1>
        Watch the segmentation loop think, call tools, and redraw masks in real
        time.
      </h1>
      <p class="viewer-subtitle">
        The viewer streams stage events, stores every overlay artifact, and lets
        you replay a completed run without tailing raw logs.
      </p>
    </div>
    <div class="viewer-hero-stats">
      <div class="metric-card">
        <span>Recent runs</span>
        <strong>{runCountLabel}</strong>
      </div>
      <div class="metric-card">
        <span>Primary use</span>
        <strong>SAM3 + Gemini critique</strong>
      </div>
    </div>
  </section>

  <section class="viewer-grid">
    <div class="panel-card launch-card">
      <div class="panel-header">
        <div>
          <p class="panel-kicker">Launch</p>
          <h2>Start a live run</h2>
        </div>
      </div>

      <form class="launch-form" on:submit|preventDefault={handleSubmit}>
        <label>
          <span>Frame URI</span>
          <textarea
            bind:value={formState.frame_uri}
            required
            rows="4"
            placeholder="gs://bucket/path/to/frame.jpg"
          ></textarea>
        </label>

        <div class="field-grid">
          <label>
            <span>Dataset name</span>
            <input bind:value={formState.dataset_name} />
          </label>
          <label>
            <span>Run label</span>
            <input bind:value={formState.run_label} />
          </label>
        </div>

        <div class="field-grid">
          <label>
            <span>Max iters</span>
            <input
              bind:value={formState.max_iters}
              type="number"
              min="1"
              max="40"
            />
          </label>
          <label>
            <span>SAM3 handler</span>
            <input bind:value={formState.sam3_handler_name} />
          </label>
        </div>

        <div class="form-actions">
          <button class="primary-button" type="submit" disabled={submitting}>
            {submitting ? 'Launching...' : 'Launch run'}
          </button>
          <button
            class="ghost-button"
            type="button"
            on:click={() => void loadRuns()}
            disabled={loading}
          >
            Refresh list
          </button>
        </div>
      </form>

      {#if error}
        <p class="error-banner">{error}</p>
      {/if}
    </div>

    <div class="panel-card">
      <div class="panel-header">
        <div>
          <p class="panel-kicker">Replay</p>
          <h2>Recent runs</h2>
        </div>
      </div>

      <div class="run-list">
        {#if loading}
          <p class="empty-state">Loading runs…</p>
        {/if}

        {#if !loading && runs.length === 0}
          <p class="empty-state">No runs recorded yet.</p>
        {/if}

        {#each runs as run (run.run_id)}
          <a class="run-card" href={resolve(`/runs/${run.run_id}`)}>
            <div class="run-card-row">
              <span class={`status-pill status-${run.status}`}>{run.status}</span>
              <span class="run-timestamp">{formatTimestamp(run.updated_at)}</span>
            </div>
            <h3>{run.run_label || run.frame_id || run.run_id}</h3>
            <p class="run-uri">{run.frame_uri}</p>
            <div class="run-meta">
              <span>{run.dataset_name || 'no dataset'}</span>
              <span>{run.sam3_handler_name || 'default SAM3'}</span>
            </div>
          </a>
        {/each}
      </div>
    </div>
  </section>
</main>
