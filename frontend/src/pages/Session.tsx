import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSSE } from "react-eventsource";
import { Link, useParams } from "react-router";
import { API_URL } from "../constants";
import { type EventSourceMessage } from "@microsoft/fetch-event-source";
import type { StreamMessageDTO } from "../types/dto";
import { useQuery } from "@tanstack/react-query";
import type { ReplayData } from "../types/replay";
import { fetchReplayData } from "../api";
import TelemetryGraphs from "../components/TelemetryGraphs";

export default function Session() {
	const params = useParams();
	const sessionID = params.session;

	if (!sessionID) {
		return <h1>Drama drama</h1>;
	}

	const [systemMessage, setSystemMessage] = useState<StreamMessageDTO>();

	const eventSourceURL = useMemo(
		() => `${API_URL}/api/sessions/${sessionID}/stream`,
		[sessionID],
	);

	const onMessage = useCallback((message: EventSourceMessage) => {
		if (!message.data) return;

		try {
			const parsedSystemMessage = JSON.parse(
				message.data,
			) as StreamMessageDTO;
			setSystemMessage(parsedSystemMessage);
		} catch (error) {
			console.warn("Ignoring non-JSON SSE message:", message.data, error);
		}
	}, []);

	const onError = useCallback((error: unknown) => {
		console.error("SSE Error:", error);
	}, []);

	const sseHeaders = useMemo(
		() => ({
			Accept: "text/event-stream",
		}),
		[],
	);

	useSSE({
		url: eventSourceURL,
		headers: sseHeaders,
		onMessage,
		onError,
	});

	const { data: replayData } = useQuery<ReplayData>({
		queryKey: ["replay_data", sessionID],
		queryFn: () => fetchReplayData(sessionID),
		enabled: systemMessage?.status === "done",
	});

	const audioSrc = useMemo(() => {
		return `data:${replayData?.audio.mime_type};base64,${replayData?.audio.base64}`;
	}, [replayData]);

	const [autoscrollEnabled, setAutoscrollEnabled] = useState(true);
	const audioRef = useRef<HTMLAudioElement>(null);
	const [currentTime, setCurrentTime] = useState(0);

	const scrollRef = useRef<HTMLLIElement>(null);

	useEffect(() => {
		if (autoscrollEnabled) {
			scrollRef.current?.scrollIntoView();
		}
	}, [scrollRef, currentTime, autoscrollEnabled]);

	const [showTelemetry, setShowTelemetry] = useState(true);
	const [showEventMarkers, setShowEventMarkers] = useState(true);

	return (
		<div className="card card-border m-5">
			<div className="card-body">
				<div className="flex items-center justify-between">
					<h2 className="card-title">{sessionID}</h2>
					<Link to="/" className="btn btn-sm">
						Back
					</Link>
				</div>

				{/* <div>Connection: {status}</div> */}

				{systemMessage?.status === "queued" && (
					<h2 className="font-semibold text-primary">
						Queued for generation
					</h2>
				)}

				<div className="w-full flex justify-center items-center">
					{systemMessage?.status === "generating_text_commentary" && (
						<div className="flex flex-col items-center gap-3">
							<div className="flex justify-center items-center gap-3">
								<div className="inline-grid *:[grid-area:1/1]">
									<div className="status status-primary animate-ping"></div>
									<div className="status status-primary"></div>
								</div>
								<h2>Commentary being generated</h2>
							</div>
							<progress
								className="progress progress-primary w-56"
								value={systemMessage.progress?.completed}
								max={systemMessage.progress?.total}
							></progress>
						</div>
					)}
					{systemMessage?.status ===
						"generating_audio_commentary" && (
						<div className="flex flex-col items-center gap-3">
							<div className="flex justify-center items-center gap-3">
								<div className="inline-grid *:[grid-area:1/1]">
									<div className="status status-secondary animate-ping"></div>
									<div className="status status-secondary"></div>
								</div>
								<h2>Synthesising audio</h2>
							</div>
							<progress
								className="progress progress-secondary w-56"
								value={systemMessage.progress?.completed}
								max={systemMessage.progress?.total}
							>
								{systemMessage.progress_text}
							</progress>
						</div>
					)}
					{systemMessage?.status === "failed" && (
						<div role="alert" className="alert alert-error">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								className="h-6 w-6 shrink-0 stroke-current"
								fill="none"
								viewBox="0 0 24 24"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth="2"
									d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
								/>
							</svg>
							<span>Failed to generate.</span>
						</div>
					)}
				</div>

				{systemMessage?.status === "done" && replayData && (
					<div className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:items-start">
						<section className="card border border-base-300 bg-base-100">
							<div className="card-body gap-4">
								<div className="flex flex-wrap items-center justify-between gap-3">
									<h3 className="font-semibold">Commentary</h3>
									<label className="label cursor-pointer gap-3">
										<span className="label-text">Autoscroll</span>
										<input
											type="checkbox"
											className="toggle"
											checked={autoscrollEnabled}
											onChange={(event) =>
												setAutoscrollEnabled(event.target.checked)
											}
										/>
									</label>
								</div>

								<ul className="list rounded-box h-72 overflow-y-auto">
									{replayData.commentary.rows.map((row) => {
										const isActive =
											row.start_time < currentTime &&
											currentTime < row.end_time;

										return (
											<li
												key={row.start_time}
												ref={isActive ? scrollRef : null}
												onClick={() => {
													if (audioRef.current) {
														audioRef.current.currentTime =
															row.start_time;
														audioRef.current.play();
													}
												}}
												className={`list-row cursor-pointer transition-all ${isActive ? "bg-accent-content" : "hover:bg-accent-content/30"}`}
											>
												<div className="flex items-center gap-4">
													<div
														className={`status status-info animate-bounce transition-all ${isActive ? "opacity-100" : "opacity-0"}`}
													></div>
													<p
														className={`transition-all ${isActive ? "font-bold" : ""}`}
													>
														{row.text}
													</p>
												</div>
											</li>
										);
									})}
								</ul>

								<div className="space-y-2">
									<h3 className="font-semibold">Playback</h3>
									<audio
										onTimeUpdate={(event) =>
											setCurrentTime(event.currentTarget.currentTime)
										}
										ref={audioRef}
										className="w-full"
										controls
										src={audioSrc}
									></audio>
								</div>
							</div>
						</section>

						<section className="card border border-base-300 bg-base-100">
							<div className="card-body gap-4">
								<div className="flex flex-wrap items-center justify-between gap-3">
									<h3 className="font-semibold">Telemetry</h3>
									<div className="flex flex-wrap items-center gap-4">
										<label className="label cursor-pointer gap-2">
											<span className="text-white">Telemetry</span>
											<input
												type="checkbox"
												className="toggle toggle-sm"
												checked={showTelemetry}
												onChange={(event) =>
													setShowTelemetry(event.target.checked)
												}
											/>
										</label>
										<label className="label cursor-pointer gap-2">
											<span className="text-white">Event markers</span>
											<input
												type="checkbox"
												className="toggle toggle-sm"
												checked={showEventMarkers}
												onChange={(event) =>
													setShowEventMarkers(event.target.checked)
												}
												disabled={!showTelemetry}
											/>
										</label>
									</div>
								</div>

								{showTelemetry ? (
									<TelemetryGraphs
										replayData={replayData}
										currentTime={currentTime}
										showEventMarkers={showEventMarkers}
									/>
								) : (
									<div className="rounded-box border border-dashed border-base-300 p-6 text-sm opacity-70">
										Telemetry graphs are hidden.
									</div>
								)}
							</div>
						</section>
					</div>
				)}
			</div>
		</div>
	);
}
