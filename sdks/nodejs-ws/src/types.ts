/**
 * Type definitions for Realm WebSocket protocol
 */

export interface FunctionCall {
  id: string;
  function: string;
  params: Record<string, any>;
  tenant_id?: string;
}

export type ResponseStatus = "streaming" | "complete" | "error" | "cancelled";

export interface ErrorInfo {
  code: string;
  message: string;
  retry_after?: number;
}

export interface FunctionResponse {
  id: string;
  status: ResponseStatus;
  data?: any;
  error?: ErrorInfo;
}

export interface GenerationOptions {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
}

export interface GenerationResult {
  text: string;
  tokens_generated: number;
  prompt_tokens?: number;
  cost_usd?: number;
  time_ms?: number;
}

export interface TokenData {
  token: string;
  index: number;
  is_final: boolean;
}

export interface PipelineInput {
  [key: string]: any;
}

export interface HealthStatus {
  status: string;
  version?: string;
  uptime_seconds?: number;
}

export interface FunctionMetadata {
  name: string;
  description: string;
  params: ParamMetadata[];
  returns: string;
  streaming: boolean;
}

export interface ParamMetadata {
  name: string;
  param_type: string;
  description: string;
  required: boolean;
  default?: any;
}

export interface RuntimeMetadata {
  version: string;
  functions: FunctionMetadata[];
}

export interface RealmClientOptions {
  /** WebSocket server URL (default: "ws://localhost:8080") */
  url?: string;
  /** API key for authentication (required for production) */
  apiKey?: string;
  /** Model name or URL (e.g., "llama-2-7b", "https://example.com/model.gguf") */
  model: string;
  /** Tenant ID - auto-assigned if not provided */
  tenantId?: string;
  /** Enable automatic reconnection (default: true) */
  reconnect?: boolean;
  /** Reconnection interval in milliseconds (default: 5000) */
  reconnectInterval?: number;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
}

export type EventCallback = (data: any) => void;
export type ErrorCallback = (error: Error) => void;

