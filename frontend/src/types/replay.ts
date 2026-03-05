export interface Session {
	session_id: string;
	created_at: string;
}

export interface ReplayData {
	session_id: string;
	status: string;
	commentary: Commentary;
	audio: Audio;
	telemetry: Telemetry;
	events: Events;
}

export interface Commentary {
	path: string;
	rows: CommentaryRow[];
	count: number;
}

export interface CommentaryRow {
	start_time: number;
	end_time: number;
	text: string;
}

export interface Audio {
	path: string;
	mime_type: string;
	bytes: number;
	base64: string;
}

export interface Telemetry {
	path: string;
	rows: TelemetryRow[];
	count: number;
}

export interface TelemetryRow {
	timestamp: string;
	current_lap: string;
	angle: string;
	current_lap_time: string;
	damage: string;
	distance_from_start: string;
	distance_raced: string;
	fuel: string;
	gear: string;
	last_lap_time: string;
	opponents: string;
	race_position: string;
	rpm: string;
	speed_x: string;
	speed_y: string;
	speed_z: string;
	distances_from_edge: string;
	distance_from_center: string;
	wheel_velocities: string;
	z: string;
	focused_distances_from_edge: string;
	dist_from_start_m: string;
	lap_progress_pct: string;
	track_length_m: string;
}

export interface Events {
	path: string;
	rows: EventRow[];
	count: number;
}

export interface EventRow {
	event_id: string;
	session_id: string;
	car_id: string;
	timestamp_s: string;
	lap: string;
	sector: string;
	dist_from_start_m: string;
	track_length_m: string;
	lap_progress_pct: string;
	track_pos: string;
	event_type: string;
	severity: string;
	confidence: string;
	speed_mps: string;
	longitudinal_accel_mps2: string;
	rpm: string;
	gear: string;
	angle_rad: string;
	lateral_speed_mps: string;
}
