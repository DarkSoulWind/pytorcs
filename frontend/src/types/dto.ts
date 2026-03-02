export type JobStatus =
	| "queued"
	| "generating_text_commentary"
	| "generating_audio_commentary"
	| "done"
	| "failed";

export type ProgressStage = string;

interface Progress {
	completed: number;
	percent: number;
	total: number;
}

export interface StreamMessageDTO {
	audio_job_id: string;
	audio_job_status: string;
	audio_path: string;
	commentary_path: string;
	pipeline_job_id: string;
	progress?: Progress;
	progress_stage?: ProgressStage;
	progress_text?: string;
	session_dir: string;
	session_id: string;
	stage: string;
	status: JobStatus;
	text_job_id: string;
	text_job_status: string;
}

export interface NewSessionDTO {
	file: File;
	name: string;
}
