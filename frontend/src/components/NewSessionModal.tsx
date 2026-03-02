import { useRef, useState } from "react";
import type { SubmitEventHandler } from "react";
import { useMutation } from "@tanstack/react-query";
import { API_URL } from "../constants";
import type { NewSessionDTO } from "../types/dto";

const getSessionName = (name: string, file: File) => {
	const trimmedName = name.trim();
	if (trimmedName.length > 0) {
		return trimmedName;
	}

	return file.name.replace(/\.csv$/i, "");
};

interface NewSessionModalProps {
	onCreated: (sessionName: string) => void | Promise<void>;
}

function NewSessionModal({ onCreated }: NewSessionModalProps) {
	const modalRef = useRef<HTMLDialogElement>(null);
	const [sessionName, setSessionName] = useState("");
	const [csvFile, setCsvFile] = useState<File | null>(null);
	const [submitError, setSubmitError] = useState<string | null>(null);

	const createSessionMutation = useMutation({
		mutationFn: async (variables: NewSessionDTO) => {
			const payload = new FormData();
			payload.append("file", variables.file);
			payload.append("name", variables.name);

			const response = await fetch(`${API_URL}/api/sessions`, {
				method: "POST",
				body: payload,
			});
			if (!response.ok) {
				throw new Error(`Failed to create session (${response.status})`);
			}

			return variables.name;
		},
		onSuccess: async (createdSessionName) => {
			modalRef.current?.close();
			setSessionName("");
			setCsvFile(null);
			setSubmitError(null);
			await onCreated(createdSessionName);
		},
		onError: (submitErr) => {
			setSubmitError((submitErr as Error).message);
		},
	});

	const handleCreateSession: SubmitEventHandler<HTMLFormElement> = async (
		event,
	) => {
		event.preventDefault();
		setSubmitError(null);

		if (!csvFile) {
			setSubmitError("Please select a CSV file.");
			return;
		}

		const isCsvFile =
			csvFile.type === "text/csv" ||
			csvFile.name.toLowerCase().endsWith(".csv");
		if (!isCsvFile) {
			setSubmitError("Only CSV files are accepted.");
			return;
		}

		const nextSessionName = getSessionName(sessionName, csvFile);
		createSessionMutation.mutate({ file: csvFile, name: nextSessionName });
	};

	return (
		<>
			<button className="btn" onClick={() => modalRef.current?.showModal()}>
				New Session
			</button>
			<dialog
				id="new-session-modal"
				className="modal modal-bottom sm:modal-middle"
				ref={modalRef}
			>
				<div className="modal-box">
					<h3 className="font-bold text-lg">Create a new session</h3>
					<form onSubmit={handleCreateSession}>
						<fieldset className="fieldset">
							<legend className="fieldset-legend">Session name</legend>
							<input
								type="text"
								className="input"
								placeholder="Type here"
								value={sessionName}
								onChange={(event) => setSessionName(event.target.value)}
							/>
							<p className="label">
								If left blank, the CSV file name will be used
							</p>
						</fieldset>

						<fieldset className="fieldset">
							<legend className="fieldset-legend">Telemetry file</legend>
							<input
								type="file"
								className="file-input"
								accept=".csv,text/csv"
								onChange={(event) => setCsvFile(event.target.files?.[0] ?? null)}
							/>
							<label className="label">
								Only CSV files with the following columns are accepted
							</label>
						</fieldset>

						{submitError && (
							<div role="alert" className="alert alert-error mt-2">
								<span>{submitError}</span>
							</div>
						)}

						<div className="modal-action">
							<button
								type="submit"
								className="btn btn-primary"
								disabled={createSessionMutation.isPending}
							>
								{createSessionMutation.isPending ? "Creating..." : "Create"}
							</button>
							<button
								type="button"
								className="btn"
								onClick={() => modalRef.current?.close()}
								disabled={createSessionMutation.isPending}
							>
								Close
							</button>
						</div>
					</form>
				</div>
			</dialog>
		</>
	);
}

export default NewSessionModal;
