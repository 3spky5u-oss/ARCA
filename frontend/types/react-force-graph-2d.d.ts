declare module 'react-force-graph-2d' {
  import { Component } from 'react';

  interface NodeObject {
    id: string | number;
    [key: string]: unknown;
  }

  interface LinkObject {
    source: string | number | NodeObject;
    target: string | number | NodeObject;
    [key: string]: unknown;
  }

  interface GraphData {
    nodes: NodeObject[];
    links: LinkObject[];
  }

  interface ForceGraph2DProps {
    graphData: GraphData;
    width?: number;
    height?: number;
    backgroundColor?: string;
    nodeLabel?: string | ((node: NodeObject) => string);
    nodeColor?: string | ((node: NodeObject) => string);
    nodeVal?: number | ((node: NodeObject) => number);
    nodeRelSize?: number;
    linkLabel?: string | ((link: LinkObject) => string);
    linkColor?: string | ((link: LinkObject) => string);
    linkWidth?: number | ((link: LinkObject) => number);
    linkDirectionalArrowLength?: number;
    linkDirectionalArrowRelPos?: number;
    onNodeClick?: (node: NodeObject) => void;
    onNodeRightClick?: (node: NodeObject) => void;
    onNodeHover?: (node: NodeObject | null) => void;
    onLinkClick?: (link: LinkObject) => void;
    onLinkRightClick?: (link: LinkObject) => void;
    onLinkHover?: (link: LinkObject | null) => void;
    cooldownTicks?: number;
    cooldownTime?: number;
    d3AlphaDecay?: number;
    d3VelocityDecay?: number;
    warmupTicks?: number;
    enableNodeDrag?: boolean;
    enableZoomInteraction?: boolean;
    enablePanInteraction?: boolean;
    dagMode?: string;
    dagLevelDistance?: number;
  }

  export default class ForceGraph2D extends Component<ForceGraph2DProps> {}
}
