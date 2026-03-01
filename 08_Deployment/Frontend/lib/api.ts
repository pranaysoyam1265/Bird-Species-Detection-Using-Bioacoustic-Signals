/**
 * BirdSense API Service
 * Handles communication with the FastAPI backend.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Prediction {
  species: string;
  scientific: string;
  confidence: number;
}

export interface DetectionSegment {
  start_time: number;
  end_time: number;
  species: string;
  scientific: string;
  confidence: number;
}

export interface DetectResponse {
  status: string;
  duration: number;
  processing_time_ms: number;
  top_species: string;
  top_scientific: string;
  top_confidence: number;
  predictions: Prediction[];
  segments: DetectionSegment[];
}

export const api = {
  /**
   * Send audio file for detection
   */
  async detect(
    audioFile: File,
    options: {
      topK?: number;
      confidenceThreshold?: number;
      noiseReduction?: boolean;
    } = {}
  ): Promise<DetectResponse> {
    const formData = new FormData();
    formData.append("audio_file", audioFile);
    if (options.topK) formData.append("top_k", options.topK.toString());
    if (options.confidenceThreshold)
      formData.append("confidence_threshold", options.confidenceThreshold.toString());
    if (options.noiseReduction !== undefined)
      formData.append("noise_reduction", options.noiseReduction.toString());

    const response = await fetch(`${API_URL}/detect`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Network error" }));
      throw new Error(error.detail || "Detection failed");
    }

    return response.json();
  },

  /**
   * Check Backend Health
   */
  async health() {
    const res = await fetch(`${API_URL}/health`);
    return res.json();
  },

  /**
   * Authentication
   */
  async login(credentials: any) {
    const res = await fetch(`${API_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(credentials),
    });
    if (!res.ok) throw new Error("Invalid credentials");
    return res.json();
  },

  async register(data: any) {
    const res = await fetch(`${API_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Registration failed");
    return res.json();
  },

  /**
   * History
   */
  async getHistory(userId: number, limit = 50, offset = 0) {
    const res = await fetch(`${API_URL}/history?user_id=${userId}&limit=${limit}&offset=${offset}`);
    if (!res.ok) throw new Error("Failed to fetch history");
    return res.json();
  },
};
