import { useEffect, useState, type ReactElement } from "react";
import { api, clearSession, storeSession, storedToken } from "./api/client";
import { BatchesView } from "./components/BatchesView";
import { ReleasesView } from "./components/ReleasesView";
import { SystemView } from "./components/SystemView";
import { ErrorBox } from "./components/primitives";

type Tab = "batches" | "releases" | "system";

export function App(): ReactElement {
  const [tab, setTab] = useState<Tab>("batches");
  const [mode, setMode] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [workspace, setWorkspace] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const info = await api.info();
        setMode(String(info["mode"] ?? "local"));
        // A session is created on first load and kept in localStorage, so a
        // reload keeps working. The API decides the workspace from the token,
        // never from anything the client sends.
        if (!storedToken()) {
          const s = await api.createSession();
          storeSession(s);
        }
        // The session works, so the one-shot 401 recovery guard can be cleared.
        sessionStorage.removeItem("forgesight.reloaded-for-auth");
        setWorkspace(localStorage.getItem("forgesight.workspace"));
        setReady(true);
      } catch (e) {
        setError(e);
        setReady(true);
      }
    })();
  }, []);

  if (!ready) {
    return (
      <div className="app">
        <div className="main">
          <p className="hint">starting…</p>
        </div>
      </div>
    );
  }

  if (error && !storedToken()) {
    return (
      <div className="app">
        <div className="main stack">
          <ErrorBox error={error} />
          <div className="panel">
            <p className="faint">
              The API did not answer. Start it with{" "}
              <span className="mono">make api</span> or{" "}
              <span className="mono">uv run forgesight.api.app:app</span>, then reload.
            </p>
            <button onClick={() => location.reload()}>retry</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {mode === "public" ? (
        <div className="banner">
          synthetic demo data · CPU · results valid only for the host and date stamped on
          each report
        </div>
      ) : null}

      <header className="topbar">
        <h1>ForgeSight</h1>
        <span className="sub">document-layout inference workbench</span>
        <div className="spacer" />
        <span className="faint mono">{workspace?.slice(0, 12) ?? ""}</span>
        <span className="pill info">{mode ?? "local"}</span>
        <button
          onClick={() => {
            clearSession();
            location.reload();
          }}
          title="Forget the stored session token"
        >
          reset session
        </button>
      </header>

      <nav className="tabs">
        {(
          [
            ["batches", "Batches"],
            ["releases", "Releases"],
            ["system", "System"],
          ] as Array<[Tab, string]>
        ).map(([id, label]) => (
          <button
            key={id}
            aria-current={tab === id}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      <main className="main">
        {tab === "batches" ? <BatchesView onChanged={() => setTab("batches")} /> : null}
        {tab === "releases" ? <ReleasesView onChanged={() => setTab("batches")} /> : null}
        {tab === "system" ? <SystemView /> : null}
        <ErrorBox error={error} />
      </main>
    </div>
  );
}
