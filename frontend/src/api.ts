import { API_URL } from "./constants";
import type { ReplayData, Session } from "./types/replay";

export async function fetchSessions(): Promise<Session[]> {
	const response = await fetch(`${API_URL}/api/sessions`);
	if (!response.ok) {
		throw new Error(`Failed to load sessions (${response.status})`);
	}
	const jsonData = await response.json();
	return jsonData.sessions ?? [];
}

export async function deleteSession(sessionName: string): Promise<void> {
	const response = await fetch(
		`${API_URL}/api/sessions/${encodeURIComponent(sessionName)}`,
		{
			method: "DELETE",
		},
	);

	if (!response.ok) {
		throw new Error(`Failed to delete session (${response.status})`);
	}
}

export async function fetchReplayData(sessionID: string): Promise<ReplayData> {
	const res = await fetch(
		`${API_URL}/api/sessions/${sessionID}/result?include_telemetry_csv=true&include_events_csv=true`,
	);

	if (!res.ok) throw new Error("Result not ready");

	const jsonData = await res.json();
	console.log("Replay data", jsonData);
	return jsonData;
}
