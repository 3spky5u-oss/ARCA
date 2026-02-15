/**
 * Domain configuration types and defaults for ARCA.
 *
 * The domain config is fetched from /api/domain on mount and provides
 * branding, tool lists, and UI content that varies by domain pack.
 */

export interface DomainConfig {
  name: string;
  display_name: string;
  app_name: string;
  tagline: string;
  primary_color: string;
  tools: string[];
  routes: string[];
  admin_visible: boolean;
  thinking_messages: string[];
  welcome_message: string;
}

export const DEFAULT_DOMAIN: DomainConfig = {
  name: 'example',
  display_name: 'ARCA',
  app_name: 'ARCA',
  tagline: 'From Documents to Domain Expert',
  primary_color: '#6366f1',
  tools: [],
  routes: [],
  admin_visible: true,
  thinking_messages: [
    'Pondering',
    'Cogitating',
    'Ruminating',
    'Mulling it over',
    'Connecting the dots',
    'Working on it',
    'Crunching the numbers',
    'Assembling the pieces',
  ],
  welcome_message: 'Hello! How can I help?',
};
