/**
 * Realm WebSocket Client
 * 
 * Connects to Realm.ai WebSocket server and provides type-safe function dispatch.
 */

import WebSocket from "ws";
import { v4 as uuidv4 } from "uuid";
import {
  FunctionCall,
  FunctionResponse,
  GenerationOptions,
  GenerationResult,
  PipelineInput,
  RealmClientOptions,
  EventCallback,
  ErrorCallback,
  TokenData,
} from "./types";

export class RealmWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private apiKey?: string;
  private tenantId: string;
  private model: string;
  private reconnect: boolean;
  private reconnectInterval: number;
  private timeout: number;
  private pendingRequests: Map<string, {
    resolve: (value: FunctionResponse) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }> = new Map();
  private streamingCallbacks: Map<string, EventCallback> = new Map();
  private isAuthenticated: boolean = false;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private eventCallbacks: Map<string, EventCallback[]> = new Map();
  private errorCallbacks: ErrorCallback[] = [];

  constructor(options: RealmClientOptions) {
    if (!options.model) {
      throw new Error("Model name or URL is required. Provide 'model' in options.");
    }

    this.url = options.url || "ws://localhost:8080";
    this.apiKey = options.apiKey;
    this.model = options.model;
    
    // Auto-assign tenant ID if not provided
    this.tenantId = options.tenantId || uuidv4();
    
    if (!options.tenantId) {
      console.log(`Auto-assigned tenant ID: ${this.tenantId}`);
    }

    this.reconnect = options.reconnect ?? true;
    this.reconnectInterval = options.reconnectInterval ?? 5000;
    this.timeout = options.timeout ?? 30000;
  }

  /**
   * Get the current tenant ID (auto-assigned or provided)
   */
  getTenantId(): string {
    return this.tenantId;
  }

  /**
   * Get the model name/URL
   */
  getModel(): string {
    return this.model;
  }

  /**
   * Connect to the WebSocket server
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      try {
        this.ws = new WebSocket(this.url);

        this.ws.on("open", () => {
          console.log("Connected to Realm server");
          this.isAuthenticated = false;
          
          // If API key is provided, authenticate
          if (this.apiKey) {
            this.authenticate(this.apiKey)
              .then(() => {
                this.isAuthenticated = true;
                this.emit("connected");
                resolve();
              })
              .catch(reject);
          } else {
            this.isAuthenticated = true;
            this.emit("connected");
            resolve();
          }
        });

        this.ws.on("message", (data: WebSocket.Data) => {
          this.handleMessage(data.toString());
        });

        this.ws.on("error", (error: Error) => {
          console.error("WebSocket error:", error);
          this.emit("error", error);
          this.errorCallbacks.forEach(cb => cb(error));
        });

        this.ws.on("close", () => {
          console.log("WebSocket connection closed");
          this.isAuthenticated = false;
          this.emit("disconnected");
          
          // Clear pending requests
          this.pendingRequests.forEach(({ reject, timeout }) => {
            clearTimeout(timeout);
            reject(new Error("Connection closed"));
          });
          this.pendingRequests.clear();

          // Attempt reconnection if enabled
          if (this.reconnect && !this.reconnectTimer) {
            this.reconnectTimer = setTimeout(() => {
              this.reconnectTimer = null;
              this.connect().catch(console.error);
            }, this.reconnectInterval);
          }
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Authenticate with API key
   */
  private async authenticate(apiKey: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        reject(new Error("WebSocket not connected"));
        return;
      }

      // Listen for authentication response
      const authHandler = (data: any) => {
        if (data.type === "auth_success" || data.status === "authenticated") {
          this.off("message", authHandler);
          resolve();
        } else if (data.type === "auth_failed" || data.error) {
          this.off("message", authHandler);
          reject(new Error(data.error?.message || "Authentication failed"));
        }
      };

      this.on("message", authHandler);

      // Send authentication request
      this.ws.send(JSON.stringify({
        type: "auth",
        api_key: apiKey,
      }));

      // Timeout after 5 seconds
      setTimeout(() => {
        this.off("message", authHandler);
        reject(new Error("Authentication timeout"));
      }, 5000);
    });
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(text: string): void {
    try {
      const message = JSON.parse(text);

      // Handle authentication responses
      if (message.type === "auth_success" || message.type === "auth_failed") {
        this.emit("message", message);
        return;
      }

      // Handle function responses
      if (message.id && this.pendingRequests.has(message.id)) {
        const { resolve, reject, timeout } = this.pendingRequests.get(message.id)!;
        clearTimeout(timeout);

        if (message.status === "error") {
          const error = new Error(message.error?.message || "Function call failed");
          (error as any).code = message.error?.code;
          (error as any).retryAfter = message.error?.retry_after;
          reject(error);
          this.pendingRequests.delete(message.id);
        } else if (message.status === "complete") {
          resolve(message);
          this.pendingRequests.delete(message.id);
        } else if (message.status === "streaming") {
          // Handle streaming response
          const callback = this.streamingCallbacks.get(message.id);
          if (callback) {
            callback(message.data);
          }
          // Don't resolve yet - wait for complete status
        } else if (message.status === "cancelled") {
          reject(new Error("Function call was cancelled"));
          this.pendingRequests.delete(message.id);
        }
      }
    } catch (error) {
      console.error("Failed to parse message:", error);
    }
  }

  /**
   * Send a function call and wait for response
   */
  private async callFunction(
    functionName: string,
    params: Record<string, any>,
    streamCallback?: (data: any) => void
  ): Promise<FunctionResponse> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected. Call connect() first.");
    }

    if (!this.isAuthenticated) {
      throw new Error("Not authenticated. Wait for connect() to complete.");
    }

    const call: FunctionCall = {
      id: uuidv4(),
      function: functionName,
      params,
      tenant_id: this.tenantId,
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(call.id);
        if (streamCallback) {
          this.streamingCallbacks.delete(call.id);
        }
        reject(new Error("Function call timeout"));
      }, this.timeout);

      this.pendingRequests.set(call.id, { resolve, reject, timeout });
      
      if (streamCallback) {
        this.streamingCallbacks.set(call.id, streamCallback);
      }

      this.ws!.send(JSON.stringify(call));
    });
  }

  /**
   * Generate text from a prompt
   */
  async generate(options: GenerationOptions): Promise<GenerationResult> {
    const response = await this.callFunction("generate", {
      prompt: options.prompt,
      model: this.model, // Include model in request
      max_tokens: options.max_tokens ?? 100,
      temperature: options.temperature ?? 0.7,
      stream: options.stream ?? false,
    });

    if (response.status === "error") {
      const error = new Error(response.error?.message || "Generation failed");
      (error as any).code = response.error?.code;
      (error as any).retryAfter = response.error?.retry_after;
      throw error;
    }

    return response.data as GenerationResult;
  }

  /**
   * Generate text with streaming
   * 
   * Yields tokens as they are generated from the server.
   * Each yielded value is a token string.
   */
  async *generateStream(options: GenerationOptions): AsyncGenerator<string, void, unknown> {
    if (!this.isConnected()) {
      throw new Error("Not connected to server. Call connect() first.");
    }

    // Queue for tokens (since we can't yield from event handler)
    const tokenQueue: Array<{ type: "token" | "done" | "error"; value?: string | Error }> = [];
    let queueResolver: ((value: { type: "token" | "done" | "error"; value?: string | Error }) => void) | null = null;

    const getNextToken = (): Promise<{ type: "token" | "done" | "error"; value?: string | Error }> => {
      if (tokenQueue.length > 0) {
        const item = tokenQueue.shift()!;
        if (queueResolver) {
          const resolver = queueResolver;
          queueResolver = null;
          resolver(item);
        }
        return Promise.resolve(item);
      }
      return new Promise((resolve) => {
        queueResolver = resolve;
      });
    };

    const pushToken = (item: { type: "token" | "done" | "error"; value?: string | Error }) => {
      if (queueResolver) {
        const resolver = queueResolver;
        queueResolver = null;
        resolver(item);
      } else {
        tokenQueue.push(item);
      }
    };

    // Use existing callFunction with streaming callback
    const streamOptions = { ...options, stream: true };
    let streamComplete = false;
    let streamError: Error | null = null;

    const streamCallback = (data: any) => {
      const tokenData = data as TokenData;
      if (tokenData && tokenData.token) {
        pushToken({ type: "token", value: tokenData.token });
      }
      if (tokenData && tokenData.is_final) {
        pushToken({ type: "done" });
        streamComplete = true;
      }
    };

    // Start the generation request with streaming callback
    const callPromise = this.callFunction("generate", {
      prompt: streamOptions.prompt,
      model: this.model,
      max_tokens: streamOptions.max_tokens ?? 100,
      temperature: streamOptions.temperature ?? 0.7,
      stream: true,
    }, streamCallback).catch((error) => {
      streamError = error;
      pushToken({ type: "error", value: error });
    });

    try {
      // Yield tokens as they arrive
      while (!streamComplete && !streamError) {
        const item = await getNextToken();

        if (item.type === "token" && item.value) {
          yield item.value as string;
        } else if (item.type === "done") {
          break;
        } else if (item.type === "error") {
          throw item.value as Error;
        }
      }

      // Wait for final response
      await callPromise;
    } catch (error) {
      // If error occurred, make sure to yield it
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(String(error));
    }
  }

  /**
   * Execute a pipeline
   */
  async executePipeline(pipelineName: string, input: PipelineInput): Promise<any> {
    const response = await this.callFunction("pipeline", {
      pipeline: pipelineName,
      input,
    });

    if (response.status === "error") {
      const error = new Error(response.error?.message || "Pipeline execution failed");
      (error as any).code = response.error?.code;
      throw error;
    }

    return response.data;
  }

  /**
   * Check server health
   */
  async health(): Promise<any> {
    const response = await this.callFunction("health", {});
    
    if (response.status === "error") {
      const error = new Error(response.error?.message || "Health check failed");
      (error as any).code = response.error?.code;
      throw error;
    }

    return response.data;
  }

  /**
   * Get runtime metadata
   */
  async metadata(): Promise<any> {
    const response = await this.callFunction("metadata", {});
    
    if (response.status === "error") {
      const error = new Error(response.error?.message || "Metadata request failed");
      (error as any).code = response.error?.code;
      throw error;
    }

    return response.data;
  }

  /**
   * Event emitter methods
   */
  on(event: string, callback: EventCallback): void {
    if (!this.eventCallbacks.has(event)) {
      this.eventCallbacks.set(event, []);
    }
    this.eventCallbacks.get(event)!.push(callback);
  }

  off(event: string, callback: EventCallback): void {
    const callbacks = this.eventCallbacks.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  private emit(event: string, data?: any): void {
    const callbacks = this.eventCallbacks.get(event);
    if (callbacks) {
      callbacks.forEach(cb => cb(data));
    }
  }

  /**
   * Add error callback
   */
  onError(callback: ErrorCallback): void {
    this.errorCallbacks.push(callback);
  }

  /**
   * Disconnect from server
   */
  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.reconnect = false;
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.pendingRequests.clear();
    this.streamingCallbacks.clear();
    this.isAuthenticated = false;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN && this.isAuthenticated;
  }
}

