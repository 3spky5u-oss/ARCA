'use client';

import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { DomainConfig, DEFAULT_DOMAIN } from './domain';
import { getApiBase } from './api';

interface DomainContextValue {
  domain: DomainConfig;
  refreshDomain: () => void;
}

const DomainContext = createContext<DomainContextValue>({
  domain: DEFAULT_DOMAIN,
  refreshDomain: () => {},
});

export function DomainProvider({ children }: { children: React.ReactNode }) {
  const [domain, setDomain] = useState<DomainConfig>(DEFAULT_DOMAIN);

  const refreshDomain = useCallback(() => {
    fetch(`${getApiBase()}/api/domain`)
      .then((res) => res.json())
      .then((data: DomainConfig) => setDomain(data))
      .catch(() => {
        // Falls back to DEFAULT_DOMAIN
      });
  }, []);

  useEffect(() => {
    refreshDomain();
  }, [refreshDomain]);

  return (
    <DomainContext.Provider value={{ domain, refreshDomain }}>
      {children}
    </DomainContext.Provider>
  );
}

export function useDomain(): DomainConfig {
  return useContext(DomainContext).domain;
}

export function useDomainRefresh(): () => void {
  return useContext(DomainContext).refreshDomain;
}
