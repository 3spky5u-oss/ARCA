'use client';

import React from 'react';

interface Session {
  file_id: string;
  filename: string;
  type: string;
  rag_chunks: number;
}

interface SessionsTabProps {
  sessions: Session[];
  onRefresh: () => void;
}

export function SessionsTab({ sessions, onRefresh }: SessionsTabProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Active Sessions</h2>
        <button
          onClick={onRefresh}
          className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] rounded-xl text-sm transition-colors"
        >
          Refresh
        </button>
      </div>

      {sessions.length > 0 ? (
        <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-[#1a1a1a]">
              <tr>
                <th className="px-4 py-2 text-left text-gray-400">File ID</th>
                <th className="px-4 py-2 text-left text-gray-400">Filename</th>
                <th className="px-4 py-2 text-left text-gray-400">Type</th>
                <th className="px-4 py-2 text-left text-gray-400">RAG Chunks</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => (
                <tr key={session.file_id} className="border-t border-[#3a3a3a]">
                  <td className="px-4 py-2 font-mono text-xs">{session.file_id}</td>
                  <td className="px-4 py-2">{session.filename}</td>
                  <td className="px-4 py-2">{session.type}</td>
                  <td className="px-4 py-2">{session.rag_chunks}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="bg-[#2a2a2a] rounded-lg p-8 text-center text-gray-400">
          No active sessions
        </div>
      )}
    </div>
  );
}
