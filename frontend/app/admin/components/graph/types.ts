/**
 * TypeScript interfaces for GraphRAG admin components
 */

// Node representation for visualization
export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
}

// Edge representation for visualization
export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, unknown>;
}

// Visualization data from API
export interface VisualizationData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
}

// Entity from entity browser
export interface GraphEntity {
  id: string;
  name: string;
  labels: string[];
  properties: Record<string, unknown>;
  relationship_count: number;
}

// Entity details with relationships
export interface EntityDetails extends GraphEntity {
  relationships: EntityRelationship[];
}

export interface EntityRelationship {
  type: string;
  direction: 'incoming' | 'outgoing';
  target_name: string | null;
  target_code: string | null;
  target_labels: string[];
  properties: Record<string, unknown>;
}

// Relationship from relationship browser
export interface GraphRelationship {
  source: string;
  source_labels: string[];
  relationship: string;
  properties: Record<string, unknown>;
  target: string;
  target_labels: string[];
}

// Graph stats from API
export interface GraphStatsData {
  total_nodes: number;
  total_relationships: number;
  node_counts: Record<string, number>;
  relationship_counts: Record<string, number>;
  average_degree: number;
  health: {
    status: string;
    url?: string;
    connected?: boolean;
    error?: string;
  };
}

// Label/type counts
export interface LabelInfo {
  label: string;
  count: number;
}

export interface RelationshipTypeInfo {
  type: string;
  count: number;
}

// Cypher query result
export interface CypherQueryResult {
  success: boolean;
  type: 'read' | 'write';
  rows?: Record<string, unknown>[];
  row_count?: number;
  summary?: Record<string, number>;
  elapsed_ms: number;
}

// Node colors for visualization (per style guide)
export const NODE_COLORS: Record<string, string> = {
  Standard: '#60A5FA',      // blue-400
  TestMethod: '#A78BFA',    // purple-400
  Concept: '#34D399',       // green-400
  Parameter: '#FBBF24',     // amber-400
  Equipment: '#F87171',     // red-400
  Chunk: '#9CA3AF',         // gray-400
  default: '#60A5FA',       // blue-400
};

export function getNodeColor(type: string): string {
  return NODE_COLORS[type] || NODE_COLORS.default;
}

// Sub-tab navigation
export type GraphSubTab = 'entities' | 'graph' | 'relationships' | 'query';
