"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { MessageSquare, Send, Bot, User, Loader2, X } from "lucide-react";
import { ChatClient, type ChatMessage } from "@/lib/api";

interface ChatPanelProps {
  fileId?: string;
  isOpen: boolean;
  onClose: () => void;
}

export function ChatPanel({ fileId, isOpen, onClose }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentResponse, setCurrentResponse] = useState("");

  const chatClientRef = useRef<ChatClient | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentResponse, scrollToBottom]);

  // Connect to WebSocket when panel opens
  useEffect(() => {
    if (isOpen && !chatClientRef.current) {
      const client = new ChatClient();
      chatClientRef.current = client;

      client.connect(
        () => setIsConnected(true),
        (error) => {
          console.error("Chat error:", error);
          setIsConnected(false);
        }
      );
    }

    return () => {
      // Don't disconnect on unmount - keep connection alive
    };
  }, [isOpen]);

  const handleSend = useCallback(() => {
    if (!input.trim() || isStreaming || !chatClientRef.current) return;

    const userMessage: ChatMessage = { role: "user", content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsStreaming(true);
    setCurrentResponse("");

    chatClientRef.current.send(input.trim(), fileId, (token, done, fullResponse) => {
      if (done) {
        if (fullResponse) {
          setMessages((prev) => [...prev, { role: "assistant", content: fullResponse }]);
        }
        setCurrentResponse("");
        setIsStreaming(false);
      } else {
        setCurrentResponse((prev) => prev + token);
      }
    });
  }, [input, isStreaming, fileId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed right-0 top-0 h-full w-full sm:w-[400px] bg-background border-l shadow-xl z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-primary" />
          <h2 className="font-semibold">AI Assistant</h2>
          {isConnected ? (
            <span className="flex items-center gap-1 text-xs text-green-600">
              <span className="w-2 h-2 bg-green-500 rounded-full" />
              Connected
            </span>
          ) : (
            <span className="flex items-center gap-1 text-xs text-yellow-600">
              <span className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
              Warming up
            </span>
          )}
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-5 w-5" />
        </Button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !currentResponse && (
          <div className="text-center text-muted-foreground py-8">
            <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Ask me about your analysis results.</p>
            <p className="text-sm mt-2">
              I can summarize exceedances, explain parameters, or answer questions.
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              "flex gap-3",
              msg.role === "user" ? "justify-end" : "justify-start"
            )}
          >
            {msg.role === "assistant" && (
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                <Bot className="h-4 w-4 text-primary" />
              </div>
            )}
            <div
              className={cn(
                "max-w-[80%] rounded-lg px-4 py-2",
                msg.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted"
              )}
            >
              <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
            </div>
            {msg.role === "user" && (
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                <User className="h-4 w-4 text-primary-foreground" />
              </div>
            )}
          </div>
        ))}

        {/* Streaming response */}
        {currentResponse && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
              <Bot className="h-4 w-4 text-primary" />
            </div>
            <div className="max-w-[80%] rounded-lg px-4 py-2 bg-muted">
              <p className="whitespace-pre-wrap text-sm">{currentResponse}</p>
              <span className="inline-block w-2 h-4 bg-primary/50 animate-pulse ml-1" />
            </div>
          </div>
        )}

        {/* Loading indicator when waiting for stream to start */}
        {isStreaming && !currentResponse && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
              <Bot className="h-4 w-4 text-primary" />
            </div>
            <div className="rounded-lg px-4 py-2 bg-muted">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your analysis..."
            disabled={isStreaming}
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-background"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming || !isConnected}
            size="icon"
          >
            {isStreaming ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

// Floating chat button
export function ChatButton({ onClick, hasResults }: { onClick: () => void; hasResults: boolean }) {
  return (
    <Button
      onClick={onClick}
      size="lg"
      className="fixed bottom-6 right-6 rounded-full shadow-lg h-14 w-14 p-0"
      disabled={!hasResults}
      title={hasResults ? "Open AI Assistant" : "Run analysis first"}
    >
      <MessageSquare className="h-6 w-6" />
    </Button>
  );
}
