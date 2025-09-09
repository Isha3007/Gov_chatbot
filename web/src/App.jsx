import React, { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://localhost:8000"; // use Vite proxy (see vite.config.js). Leave empty to use relative /api

export default function App() {
  const [messages, setMessages] = useState([]); // {role: 'user'|'assistant', content, sources?}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  const send = async () => {
    const q = input.trim();
    if (!q) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      setMessages((m) => [
        ...m,
        { role: "assistant", content: data.answer, sources: data.sources || [] },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `⚠️ Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // JSX UI
  return (
    <div className="h-screen flex flex-col bg-white">
      {/* Messages list */}
      <div ref={listRef} className="flex-1 overflow-y-auto p-4 space-y-2">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-2 rounded-xl max-w-[80%] ${
              msg.role === "user"
                ? "ml-auto bg-green-100 text-green-900"
                : "mr-auto bg-orange-100 text-orange-900"
            }`}
          >
            {msg.content}
            {msg.sources?.length > 0 && (
              <div className="mt-1 text-xs text-blue-600">
                {msg.sources.map((s, j) => (
                  <span
                    key={j}
                    className="inline-block mr-2 px-2 py-0.5 bg-blue-100 rounded-full"
                  >
                    {s}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="italic text-gray-500">Thinking…</div>}
      </div>

      {/* Input bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (canSend) send();
        }}
        className="p-2 border-t flex"
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={1}
          placeholder="Ask a question..."
          className="flex-1 resize-none border rounded-lg p-2 mr-2"
        />
        <button
          type="submit"
          disabled={!canSend}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
