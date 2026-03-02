import { useMemo } from "react";
import type { EventRow, ReplayData } from "../types/replay";
import ReactECharts, { type EChartsOption } from "echarts-for-react";

interface TelemetryGraphsProps {
	replayData: ReplayData;
	currentTime: number;
	showEventMarkers: boolean;
}

interface EventMarker {
	xAxis: number;
	name: string;
	label: {
		formatter: string;
		position: "insideEndTop";
		color: string;
		fontWeight: number;
	};
	lineStyle: {
		type: "dashed";
		width: number;
		color: string;
	};
}

const EVENT_COLOR_MAP: Record<string, string> = {
	LAP_START: "#1d4ed8",
	LAP_COMPLETE: "#1e40af",
	HARD_BRAKING: "#dc2626",
	STRONG_ACCELERATION: "#16a34a",
	LIFT_AND_COAST: "#65a30d",
	UPSHIFT: "#22c55e",
	DOWNSHIFT: "#f59e0b",
	OVER_REV: "#f97316",
	BOUNCING_LIMITER: "#fb7185",
	CORNER_ENTRY: "#0ea5e9",
	CORNER_EXIT: "#06b6d4",
	TRACK_LIMITS_WARNING: "#ca8a04",
	OFF_TRACK: "#e11d48",
	RUNNING_WIDE: "#f43f5e",
	SLIDE: "#a855f7",
	SPIN: "#9333ea",
	RECOVERY: "#14b8a6",
	CAR_AHEAD_CLOSE: "#2563eb",
	CAR_BEHIND_CLOSE: "#7c3aed",
	OVERTAKE: "#84cc16",
	BEING_OVERTAKEN: "#ef4444",
	SIDE_BY_SIDE: "#0891b2",
	WALL_IMPACT: "#991b1b",
	CONTACT: "#b45309",
	DAMAGE_EVENT: "#b91c1c",
	STOPPED: "#374151",
	STUCK: "#111827",
};

function getEventColor(eventType: string): string {
	const normalizedType = eventType.trim().toUpperCase();
	return EVENT_COLOR_MAP[normalizedType] ?? "#6366f1";
}

function toEventMarker(event: EventRow): EventMarker | null {
	const time = parseFloat(event.timestamp_s);
	if (!Number.isFinite(time)) return null;
	const eventType = event.event_type || "EVENT";
	const color = getEventColor(eventType);

	return {
		xAxis: time,
		name: eventType,
		label: {
			formatter: eventType,
			position: "insideEndTop",
			color,
			fontWeight: 700,
		},
		lineStyle: {
			type: "dashed",
			width: 1,
			color,
		},
	};
}

function TelemetryGraphs({
	replayData,
	currentTime,
	showEventMarkers,
}: TelemetryGraphsProps) {
	const chartWindowSizeSeconds = 10;

	const telemetrySeries = useMemo(() => {
		if (!replayData) return null;

		const timestamps = replayData.telemetry.rows.map((row) =>
			parseFloat(row.timestamp),
		);
		const rpm = replayData.telemetry.rows.map((row) => parseFloat(row.rpm));
		const speed = replayData.telemetry.rows.map((row) =>
			Math.sqrt(
				Math.pow(parseFloat(row.speed_x), 2) +
					Math.pow(parseFloat(row.speed_y), 2) +
					Math.pow(parseFloat(row.speed_z), 2),
			),
		);

		return { timestamps, rpm, speed };
	}, [replayData]);

	const chartRange = useMemo(() => {
		if (!telemetrySeries || telemetrySeries.timestamps.length === 0) {
			return { min: 0, max: chartWindowSizeSeconds };
		}

		const telemetryEnd =
			telemetrySeries.timestamps[telemetrySeries.timestamps.length - 1];
		const boundedTime = Math.min(currentTime, telemetryEnd);
		const halfWindow = chartWindowSizeSeconds / 2;
		const maxWindowStart = Math.max(
			0,
			telemetryEnd - chartWindowSizeSeconds,
		);
		const min = Math.max(
			0,
			Math.min(boundedTime - halfWindow, maxWindowStart),
		);
		const max = Math.min(telemetryEnd, min + chartWindowSizeSeconds);

		return { min, max };
	}, [telemetrySeries, currentTime]);

	const chartData = useMemo(() => {
		if (!telemetrySeries) return null;

		const { timestamps, rpm, speed } = telemetrySeries;
		const visibleTimestamps: number[] = [];
		const visibleRPM: number[] = [];
		const visibleSpeed: number[] = [];

		for (let i = 0; i < timestamps.length; i++) {
			const timestamp = timestamps[i];
			if (timestamp < chartRange.min || timestamp > chartRange.max)
				continue;
			visibleTimestamps.push(timestamp);
			visibleRPM.push(rpm[i]);
			visibleSpeed.push(speed[i]);
		}

		return {
			timestamps: visibleTimestamps,
			rpm: visibleRPM,
			speed: visibleSpeed,
		};
	}, [telemetrySeries, chartRange]);

	const visibleEventMarkers = useMemo<EventMarker[]>(() => {
		return replayData.events.rows
			.map(toEventMarker)
			.filter((marker): marker is EventMarker => marker !== null)
			.filter(
				(marker) =>
					marker.xAxis >= chartRange.min &&
					marker.xAxis <= chartRange.max,
			);
	}, [replayData.events.rows, chartRange]);

	const baseXAxis = useMemo(
		() => ({
			type: "value" as const,
			name: "Time (s)",
			min: chartRange.min,
			max: chartRange.max,
			nameTextStyle: { color: "#ffffff" },
			axisLine: { lineStyle: { color: "#ffffff" } },
			axisTick: { lineStyle: { color: "#ffffff" } },
			axisLabel: { formatter: "{value}s", color: "#ffffff" },
			splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.25)" } },
		}),
		[chartRange],
	);

	const baseYAxis = useMemo(
		() => ({
			type: "value" as const,
			nameTextStyle: { color: "#ffffff" },
			axisLine: { lineStyle: { color: "#ffffff" } },
			axisTick: { lineStyle: { color: "#ffffff" } },
			axisLabel: { color: "#ffffff" },
			splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.25)" } },
		}),
		[],
	);

	const rpmOption = useMemo<EChartsOption>(
		() => ({
			title: { text: "RPM", textStyle: { color: "#ffffff" } },
			animation: false,
			tooltip: {
				trigger: "axis",
				valueFormatter: (value: string | number) =>
					`${Number(value).toFixed(0)} rpm`,
			},
			grid: { left: 50, right: 20, top: 40, bottom: 40 },
			xAxis: baseXAxis,
			yAxis: { ...baseYAxis, name: "RPM" },
			series: [
				{
					type: "line",
					showSymbol: false,
					data:
						chartData?.timestamps.map((t, i) => [
							t,
							chartData.rpm[i],
						]) ?? [],
					lineStyle: { width: 2, color: "#ffffff" },
					markLine: {
						symbol: ["none", "none"],
						data: showEventMarkers ? visibleEventMarkers : [],
					},
				},
			],
		}),
		[
			baseXAxis,
			baseYAxis,
			chartData,
			showEventMarkers,
			visibleEventMarkers,
		],
	);

	const speedOption = useMemo<EChartsOption>(
		() => ({
			title: { text: "Speed", textStyle: { color: "#ffffff" } },
			animation: false,
			tooltip: {
				trigger: "axis",
				valueFormatter: (value: string | number) =>
					`${Number(value).toFixed(2)} m/s`,
			},
			grid: { left: 50, right: 20, top: 40, bottom: 40 },
			xAxis: baseXAxis,
			yAxis: { ...baseYAxis, name: "Speed (m/s)" },
			series: [
				{
					type: "line",
					showSymbol: false,
					data:
						chartData?.timestamps.map((t, i) => [
							t,
							chartData.speed[i],
						]) ?? [],
					lineStyle: { width: 2, color: "#ffffff" },
					markLine: {
						symbol: ["none", "none"],
						data: showEventMarkers ? visibleEventMarkers : [],
					},
				},
			],
		}),
		[
			baseXAxis,
			baseYAxis,
			chartData,
			showEventMarkers,
			visibleEventMarkers,
		],
	);

	return (
		<div className="grid grid-cols-1 3xl:grid-cols-2 gap-2">
			<ReactECharts
				className="rounded-box border border-base-300 bg-base-100 p-4"
				option={rpmOption}
				style={{ height: 320, width: "100%" }}
			/>

			<ReactECharts
				className="rounded-box border border-base-300 bg-base-100 p-4"
				option={speedOption}
				style={{ height: 320, width: "100%" }}
			/>
		</div>
	);
}

export default TelemetryGraphs;
