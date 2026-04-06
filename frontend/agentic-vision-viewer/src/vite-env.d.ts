/// <reference types="@sveltejs/kit" />

interface ImportMetaEnv {
  readonly NEXT_PUBLIC_AGENTIC_VISION_VIEWER_API_BASE_URL?: string;
  readonly PUBLIC_AGENTIC_VISION_VIEWER_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
