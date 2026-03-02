import { createRoot } from "react-dom/client";
import "./index.css";
import Home from "./pages/Home.tsx";
import Session from "./pages/Session.tsx";
import { BrowserRouter, Routes, Route } from "react-router";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
	<QueryClientProvider client={queryClient}>
		<ReactQueryDevtools />
		<BrowserRouter>
			<Routes>
				<Route index element={<Home />} />

				<Route path="sessions">
					<Route path=":session" element={<Session />} />
				</Route>
			</Routes>
		</BrowserRouter>
	</QueryClientProvider>,
);
