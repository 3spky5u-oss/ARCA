'use client';

import React from 'react';

interface ExceedeeDBStats {
  total_records: number;
  pending_count: number;
  approved_count: number;
  rejected_count: number;
  last_updated: string | null;
  last_backup: string | null;
  backup_count: number;
  projects: string[];
}

interface ExceedeeRecord {
  record_id: string;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  approved_at: string | null;
  rejected_at: string | null;
  rejection_reason: string | null;
  project_number: string;
  project_name: string | null;
  client_name: string | null;
  location_name: string;
  sample_id: string;
  sample_date: string | null;
  depth: string | null;
  lab_report_filename: string;
  exceedance_count: number;
  soil_type: string | null;
  land_use: string | null;
}

interface ExceedeeDBTabProps {
  stats: ExceedeeDBStats | null;
  pendingRecords: ExceedeeRecord[];
  browseRecords: ExceedeeRecord[];
  projectFilter: string;
  onProjectFilterChange: (filter: string) => void;
  statusFilter: string;
  onStatusFilterChange: (filter: string) => void;
  onRefresh: () => void;
  onExportRecords: (format: 'csv' | 'json') => void;
  onApproveRecord: (recordId: string) => void;
  onRejectRecord: (recordId: string) => void;
  onDeleteRecord: (recordId: string) => void;
  onSearch: () => void;
}

export function ExceedeeDBTab({
  stats,
  pendingRecords,
  browseRecords,
  projectFilter,
  onProjectFilterChange,
  statusFilter,
  onStatusFilterChange,
  onRefresh,
  onExportRecords,
  onApproveRecord,
  onRejectRecord,
  onDeleteRecord,
  onSearch,
}: ExceedeeDBTabProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">ExceedeeDB - Sample Records</h2>
        <div className="flex gap-2">
          <button
            onClick={() => onExportRecords('csv')}
            className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] rounded-xl text-sm transition-colors"
          >
            Export CSV
          </button>
          <button
            onClick={() => onExportRecords('json')}
            className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] rounded-xl text-sm transition-colors"
          >
            Export JSON
          </button>
          <button
            onClick={onRefresh}
            className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] rounded-xl text-sm transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-5 gap-4">
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-blue-400">{stats.total_records}</p>
            <p className="text-xs text-gray-400 mt-1">Total Records</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-amber-400">{stats.pending_count}</p>
            <p className="text-xs text-gray-400 mt-1">Pending</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-green-400">{stats.approved_count}</p>
            <p className="text-xs text-gray-400 mt-1">Approved</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-red-400">{stats.rejected_count}</p>
            <p className="text-xs text-gray-400 mt-1">Rejected</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-lg font-bold text-gray-400">{stats.last_backup || 'Never'}</p>
            <p className="text-xs text-gray-400 mt-1">Last Backup</p>
          </div>
        </div>
      )}

      {/* Pending Queue */}
      <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
        <div className="px-4 py-3 bg-[#1a1a1a] border-b border-[#3a3a3a] flex items-center justify-between">
          <h3 className="text-sm font-medium">Pending Approval ({pendingRecords.length})</h3>
        </div>
        {pendingRecords.length > 0 ? (
          <div className="divide-y divide-[#3a3a3a]">
            {pendingRecords.map((record) => (
              <div key={record.record_id} className="px-4 py-3 flex items-center justify-between hover:bg-[#333]">
                <div className="flex-1 grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-xs text-gray-500">Project</span>
                    <p className="font-medium">{record.project_number}</p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Sample</span>
                    <p className="font-medium">{record.sample_id}</p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Location</span>
                    <p className="text-gray-400">{record.location_name || '-'}</p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Exceedances</span>
                    <p className={record.exceedance_count > 0 ? 'text-red-400' : 'text-green-400'}>
                      {record.exceedance_count}
                    </p>
                  </div>
                </div>
                <div className="flex gap-2 ml-4">
                  <button
                    onClick={() => onApproveRecord(record.record_id)}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded-xl text-xs transition-colors"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => onRejectRecord(record.record_id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded-xl text-xs transition-colors"
                  >
                    Reject
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500">
            No pending records
          </div>
        )}
      </div>

      {/* Browse Records */}
      <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
        <div className="px-4 py-3 bg-[#1a1a1a] border-b border-[#3a3a3a]">
          <h3 className="text-sm font-medium mb-3">Browse Records</h3>
          <div className="flex gap-4">
            <div className="flex-1">
              <input
                type="text"
                value={projectFilter}
                onChange={(e) => onProjectFilterChange(e.target.value)}
                placeholder="Filter by project..."
                className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => onStatusFilterChange(e.target.value)}
              className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
            >
              <option value="">All Status</option>
              <option value="pending">Pending</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
            </select>
            <button
              onClick={onSearch}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm transition-colors"
            >
              Search
            </button>
          </div>
        </div>
        {browseRecords.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-[#1a1a1a]">
                <tr>
                  <th className="px-4 py-2 text-left text-gray-400">Project</th>
                  <th className="px-4 py-2 text-left text-gray-400">Sample</th>
                  <th className="px-4 py-2 text-left text-gray-400">Location</th>
                  <th className="px-4 py-2 text-left text-gray-400">Status</th>
                  <th className="px-4 py-2 text-left text-gray-400">Exceedances</th>
                  <th className="px-4 py-2 text-left text-gray-400">Created</th>
                  <th className="px-4 py-2 text-left text-gray-400">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#3a3a3a]">
                {browseRecords.map((record) => (
                  <tr key={record.record_id} className="hover:bg-[#333]">
                    <td className="px-4 py-2">{record.project_number}</td>
                    <td className="px-4 py-2">{record.sample_id}</td>
                    <td className="px-4 py-2 text-gray-400">{record.location_name || '-'}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-0.5 rounded-full text-xs ${
                        record.status === 'approved' ? 'bg-green-600/20 text-green-400' :
                        record.status === 'pending' ? 'bg-amber-600/20 text-amber-400' :
                        'bg-red-600/20 text-red-400'
                      }`}>
                        {record.status}
                      </span>
                    </td>
                    <td className={`px-4 py-2 ${record.exceedance_count > 0 ? 'text-red-400' : 'text-green-400'}`}>
                      {record.exceedance_count}
                    </td>
                    <td className="px-4 py-2 text-gray-400 text-xs">
                      {record.created_at ? new Date(record.created_at).toLocaleDateString() : '-'}
                    </td>
                    <td className="px-4 py-2">
                      <button
                        onClick={() => onDeleteRecord(record.record_id)}
                        className="text-red-400 hover:text-red-300 text-xs"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500">
            No records found
          </div>
        )}
      </div>

      {/* Projects List */}
      {stats && stats.projects.length > 0 && (
        <div className="bg-[#2a2a2a] rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3">Projects ({stats.projects.length})</h3>
          <div className="flex flex-wrap gap-2">
            {stats.projects.map((project) => (
              <button
                key={project}
                onClick={() => { onProjectFilterChange(project); onSearch(); }}
                className="px-3 py-1 bg-[#1a1a1a] hover:bg-[#333] rounded-xl text-xs transition-colors"
              >
                {project}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
