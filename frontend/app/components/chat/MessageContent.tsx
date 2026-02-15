'use client';

import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import { CodeBlock } from './CodeBlock';

export interface MessageContentProps {
  content: string;
  isUser?: boolean;
}

// Memoized plugins to prevent re-creation on each render
const remarkPlugins = [remarkMath, remarkGfm];
const rehypePlugins = [rehypeKatex];

function MessageContentInner({ content, isUser }: MessageContentProps) {
  // Memoize components object to prevent re-creation
  const components = useMemo(() => ({
    code({ className, children, ...props }: { className?: string; children?: React.ReactNode }) {
      const match = /language-(\w+)/.exec(className || '');
      const isInline = !match && !className;

      if (isInline) {
        return (
          <code className="inline-code" {...props}>
            {children}
          </code>
        );
      }

      return (
        <CodeBlock language={match?.[1] || ''}>
          {String(children).replace(/\n$/, '')}
        </CodeBlock>
      );
    },
    p({ children }: { children?: React.ReactNode }) {
      return <p className="mb-2 last:mb-0">{children}</p>;
    },
    ul({ children }: { children?: React.ReactNode }) {
      return <ul className="list-disc pl-6 my-2 space-y-1">{children}</ul>;
    },
    ol({ children }: { children?: React.ReactNode }) {
      return <ol className="list-decimal pl-6 my-2 space-y-1">{children}</ol>;
    },
    li({ children }: { children?: React.ReactNode }) {
      return <li className="leading-relaxed">{children}</li>;
    },
    a({ href, children }: { href?: string; children?: React.ReactNode }) {
      return (
        <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 underline">
          {children}
        </a>
      );
    },
  }), []);

  if (!content) return null;

  return (
    <div className={`markdown-content ${isUser ? 'text-white' : ''}`}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export const MessageContent = React.memo(MessageContentInner);
MessageContent.displayName = 'MessageContent';
