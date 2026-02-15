'use client';

import React from 'react';

interface SubTab {
  id: string;
  label: string;
}

interface SubTabLayoutProps {
  tabs: SubTab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  children: React.ReactNode;
}

export function SubTabLayout({ tabs, activeTab, onTabChange, children }: SubTabLayoutProps) {
  return (
    <div>
      <div className="flex gap-1 border-b border-[#3a3a3a] mb-6">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === tab.id
                ? 'text-blue-400 border-blue-400'
                : 'text-gray-400 border-transparent hover:text-gray-200 hover:border-gray-600'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {children}
    </div>
  );
}
