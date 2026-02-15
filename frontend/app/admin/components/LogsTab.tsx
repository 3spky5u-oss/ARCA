'use client';

import React from 'react';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  module: string;
}

interface LogsTabProps {
  logs: LogEntry[];
  onRefresh: () => void;
}

export function LogsTab({ logs, onRefresh }: LogsTabProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Recent Logs</h2>
        <button
          onClick={onRefresh}
          className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] rounded-xl text-sm transition-colors"
        >
          Refresh
        </button>
      </div>

      <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
        <div className="max-h-[600px] overflow-auto">
          <table className="w-full text-xs font-mono">
            <thead className="bg-[#1a1a1a] sticky top-0">
              <tr>
                <th className="px-3 py-2 text-left text-gray-400 w-40">Time</th>
                <th className="px-3 py-2 text-left text-gray-400 w-20">Level</th>
                <th className="px-3 py-2 text-left text-gray-400 w-24">Module</th>
                <th className="px-3 py-2 text-left text-gray-400">Message</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, i) => (
                <tr key={i} className="border-t border-[#3a3a3a] hover:bg-[#333]">
                  <td className="px-3 py-1 text-gray-500">{log.timestamp.split('T')[1]?.split('.')[0]}</td>
                  <td className="px-3 py-1">
                    <span
                      className={`${
                        log.level === 'ERROR'
                          ? 'text-red-400'
                          : log.level === 'WARNING'
                          ? 'text-yellow-400'
                          : 'text-gray-400'
                      }`}
                    >
                      {log.level}
                    </span>
                  </td>
                  <td className="px-3 py-1 text-gray-500">{log.module}</td>
                  <td className="px-3 py-1 text-gray-300 truncate max-w-xl">{log.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
