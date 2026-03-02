import { useRef, useState } from "react";
import { useNavigate } from "react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import NewSessionModal from "../components/NewSessionModal";
import type { Session } from "../types/replay";
import { deleteSession, fetchSessions } from "../api";

function Home() {
	const navigate = useNavigate();
	const queryClient = useQueryClient();
	const deleteModalRef = useRef<HTMLDialogElement>(null);
	const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
	const [deleteError, setDeleteError] = useState<string | null>(null);
	const {
		data: sessions = [],
		isLoading,
		isError,
		error,
	} = useQuery<Session[]>({
		queryKey: ["sessions"],
		queryFn: fetchSessions,
	});

	const deleteSessionMutation = useMutation({
		mutationFn: deleteSession,
		onSuccess: async () => {
			deleteModalRef.current?.close();
			setSessionToDelete(null);
			setDeleteError(null);
			await queryClient.invalidateQueries({ queryKey: ["sessions"] });
		},
		onError: (error) => {
			setDeleteError((error as Error).message);
		},
	});

	const handleSessionCreated = async (createdSessionName: string) => {
		await queryClient.invalidateQueries({ queryKey: ["sessions"] });
		navigate(
			`/sessions/${encodeURIComponent(createdSessionName.split(" ").join("_"))}`,
		);
	};

	const openDeleteModal = (sessionName: string) => {
		setSessionToDelete(sessionName);
		setDeleteError(null);
		deleteModalRef.current?.showModal();
	};

	const handleDeleteConfirm = () => {
		if (!sessionToDelete) {
			return;
		}

		deleteSessionMutation.mutate(sessionToDelete);
	};

	if (isLoading) {
		return <span className="loading loading-spinner loading-lg"></span>;
	}

	if (isError) {
		return (
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
				<span>Sessions failed to load: {(error as Error).message}</span>
			</div>
		);
	}

	return (
		<div className="card card-border m-5">
			<div className="card-body">
				<h2 className="card-title">Your sessions</h2>

				<ul className="list bg-base-100 rounded-box">
					{!sessions.length && (
						<p>
							Click{" "}
							<span className="font-semibold">New Session</span>{" "}
							to begin generating commentary.
						</p>
					)}
					{sessions.map((session) => (
						<li
							className="list-row transition-all hover:bg-accent-content hover:cursor-pointer"
							onClick={() =>
								navigate(`/sessions/${session.session_id}`)
							}
							key={session.session_id}
						>
							<div></div>
							<div>
								<div>{session.session_id}</div>
								<div className="text-xs uppercase font-semibold opacity-60">
									{new Date(
										session.created_at,
									).toLocaleString()}
								</div>
							</div>
							<div>
								<button
									type="button"
									className="btn btn-ghost btn-xs text-error"
									onClick={(event) => {
										event.stopPropagation();
										openDeleteModal(session.session_id);
									}}
								>
									Delete
								</button>
							</div>
						</li>
					))}
				</ul>

				<div className="card-actions justify-end">
					<NewSessionModal onCreated={handleSessionCreated} />
				</div>
			</div>

			<dialog
				id="delete-session-modal"
				className="modal modal-bottom sm:modal-middle"
				ref={deleteModalRef}
			>
				<div className="modal-box">
					<h3 className="font-bold text-lg">Delete session</h3>
					<p className="py-4">
						Are you sure you want to delete{" "}
						<span className="font-semibold">{sessionToDelete}</span>?
					</p>

					{deleteError && (
						<div role="alert" className="alert alert-error mt-2">
							<span>{deleteError}</span>
						</div>
					)}

					<div className="modal-action">
						<button
							type="button"
							className="btn btn-error"
							onClick={handleDeleteConfirm}
							disabled={deleteSessionMutation.isPending || !sessionToDelete}
						>
							{deleteSessionMutation.isPending ? "Deleting..." : "Delete"}
						</button>
						<button
							type="button"
							className="btn"
							onClick={() => {
								deleteModalRef.current?.close();
								setSessionToDelete(null);
								setDeleteError(null);
							}}
							disabled={deleteSessionMutation.isPending}
						>
							Cancel
						</button>
					</div>
				</div>
			</dialog>
		</div>
	);
}

export default Home;
