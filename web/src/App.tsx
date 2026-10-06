import { useEffect, useState, type ReactElement } from "react";
import { Layers, RotateCcw, AlertCircle, Sparkles } from "lucide-react";
import { api, clearSession, storeSession, storedToken } from "./api/client";
import { BatchesView } from "./components/BatchesView";
import { ReleasesView } from "./components/ReleasesView";
import { SystemView } from "./components/SystemView";
import { ErrorBox } from "./components/primitives";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "./components/ui/card";
import { cn } from "./lib/utils";

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
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="flex items-center gap-2.5 text-zinc-400 text-sm">
          <Sparkles className="size-4 animate-spin text-sky-400" />
          <span>Starting ForgeSight…</span>
        </div>
      </div>
    );
  }

  if (error && !storedToken()) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-6">
        <Card className="max-w-md w-full border-red-500/20 bg-zinc-900/90 shadow-2xl">
          <CardHeader className="flex flex-row items-center gap-3 space-y-0 pb-3">
            <div className="flex size-9 items-center justify-center rounded-lg bg-red-500/10 text-red-400 border border-red-500/20 shrink-0">
              <AlertCircle className="size-5" />
            </div>
            <div>
              <CardTitle className="text-base text-zinc-100">Connection Failed</CardTitle>
              <CardDescription className="text-xs text-zinc-400">
                Unable to establish connection to ForgeSight API service
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 pt-2">
            <ErrorBox error={error} />
          </CardContent>
          <CardFooter className="flex justify-end pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => location.reload()}
              className="gap-2 text-xs"
            >
              <RotateCcw className="size-3.5" />
              Retry
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  const isLive = mode === "public" || mode === "live";
  const modeLabel = isLive ? "Live" : "Local";

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <Tabs
        value={tab}
        onValueChange={(v) => setTab(v as Tab)}
        className="flex flex-col flex-1 min-h-0 gap-0"
      >
        <header className="topbar flex items-center justify-between border-b border-border bg-card/80 px-4 py-2.5 backdrop-blur-sm gap-4">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2.5">
              <div className="flex size-7 items-center justify-center rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20 shadow-sm">
                <Layers className="size-4" />
              </div>
              <h1 className="text-sm font-semibold tracking-tight text-foreground m-0">
                ForgeSight
              </h1>
            </div>

            <TabsList className="bg-zinc-950/60 border border-border">
              <TabsTrigger value="batches">Batches</TabsTrigger>
              <TabsTrigger value="releases">Releases</TabsTrigger>
              <TabsTrigger value="system">System</TabsTrigger>
            </TabsList>
          </div>

          <div className="flex items-center gap-2.5">
            {workspace ? (
              <Badge
                variant="outline"
                className="font-mono text-xs text-muted-foreground border-border bg-zinc-900/50"
              >
                <span className="text-zinc-500 mr-1.5">ws:</span>
                {workspace.slice(0, 12)}
              </Badge>
            ) : null}
            <Badge variant={isLive ? "success" : "secondary"} className="text-xs font-medium">
              <span
                className={cn(
                  "size-1.5 rounded-full mr-1.5",
                  isLive ? "bg-emerald-400" : "bg-zinc-400"
                )}
              />
              {modeLabel}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                clearSession();
                location.reload();
              }}
              title="Forget the stored session token"
              className="gap-1.5 text-xs text-zinc-300 hover:text-zinc-100"
            >
              <RotateCcw className="size-3" />
              Reset
            </Button>
          </div>
        </header>

        <main className="main flex-1 overflow-auto p-4">
          <TabsContent value="batches" className="m-0 focus-visible:ring-0">
            <BatchesView onChanged={() => setTab("batches")} />
          </TabsContent>
          <TabsContent value="releases" className="m-0 focus-visible:ring-0">
            <ReleasesView onChanged={() => setTab("batches")} />
          </TabsContent>
          <TabsContent value="system" className="m-0 focus-visible:ring-0">
            <SystemView />
          </TabsContent>
          <ErrorBox error={error} />
        </main>
      </Tabs>
    </div>
  );
}

