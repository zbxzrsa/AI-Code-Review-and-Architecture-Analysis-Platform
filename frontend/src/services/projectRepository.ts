export interface ProjectRecord {
  id: string;
  name: string;
  description: string;
  status: 'Active' | 'On Hold' | 'Archived';
  owner: string;
  repoUrl?: string;
  tags: string[];
  lastAnalysis?: string;
  qualityScore: number;
  riskScore: number;
  createdAt: string;
  updatedAt: string;
  demoLink?: string;
}

export interface ProjectActivity {
  id: string;
  projectId: string;
  title: string;
  description: string;
  type: 'analysis' | 'session' | 'release' | 'alert';
  actor: string;
  timestamp: string;
}

const PROJECT_STORAGE_KEY = 'icraap.projects';
const ACTIVITY_STORAGE_KEY = 'icraap.projectActivities';

const defaultProjects: ProjectRecord[] = [
  {
    id: 'proj-001',
    name: 'E-Commerce Experience',
    description:
      'Monorepo for storefront, payment integrations, and internal operations portals.',
    status: 'Active',
    owner: 'Jessie Wilder',
    repoUrl: 'https://github.com/example/ecommerce-experience',
    tags: ['web', 'payments', 'react'],
    lastAnalysis: new Date().toISOString(),
    qualityScore: 82,
    riskScore: 18,
    createdAt: '2024-08-12T07:00:00.000Z',
    updatedAt: new Date().toISOString(),
    demoLink: 'https://demo.example.com/ecommerce',
  },
  {
    id: 'proj-002',
    name: 'Observability Gateway',
    description:
      'Edge gateway that aggregates traces, metrics, and logs from hybrid clusters.',
    status: 'Active',
    owner: 'Nikhil Patel',
    repoUrl: 'https://github.com/example/observability-gateway',
    tags: ['go', 'otel', 'security'],
    lastAnalysis: '2024-11-05T14:35:00.000Z',
    qualityScore: 90,
    riskScore: 12,
    createdAt: '2024-05-01T10:30:00.000Z',
    updatedAt: '2024-11-05T14:40:00.000Z',
    demoLink: 'https://demo.example.com/otel',
  },
  {
    id: 'proj-003',
    name: 'Mobile Banking Toolkit',
    description:
      'Shared components, security policies, and CI/CD templates for the mobile banking suite.',
    status: 'On Hold',
    owner: 'Aisha Gomez',
    repoUrl: 'https://github.com/example/mobile-banking-toolkit',
    tags: ['mobile', 'security', 'compliance'],
    lastAnalysis: '2024-10-02T09:00:00.000Z',
    qualityScore: 68,
    riskScore: 32,
    createdAt: '2023-11-21T11:00:00.000Z',
    updatedAt: '2024-10-02T09:05:00.000Z',
    demoLink: undefined,
  },
];

const defaultActivities: ProjectActivity[] = [
  {
    id: 'act-001',
    projectId: 'proj-001',
    title: 'Architecture session completed',
    description: 'Identified tight coupling between cart service and pricing SDK.',
    type: 'analysis',
    actor: 'Automated Review Bot',
    timestamp: new Date().toISOString(),
  },
  {
    id: 'act-002',
    projectId: 'proj-001',
    title: 'Baseline drift alert',
    description: 'Maintainability index dropped by 6 points compared to baseline.',
    type: 'alert',
    actor: 'Baseline Guardian',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 6).toISOString(),
  },
  {
    id: 'act-003',
    projectId: 'proj-002',
    title: 'Release candidate approved',
    description: 'Version 2.4.0 passed regression tests with zero blocking issues.',
    type: 'release',
    actor: 'Nikhil Patel',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 26).toISOString(),
  },
];

const load = <T,>(key: string, fallback: T): T => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
};

const save = <T,>(key: string, data: T): void => {
  localStorage.setItem(key, JSON.stringify(data));
};

const ensureSeedData = (): void => {
  if (!localStorage.getItem(PROJECT_STORAGE_KEY)) {
    save(PROJECT_STORAGE_KEY, defaultProjects);
  }
  if (!localStorage.getItem(ACTIVITY_STORAGE_KEY)) {
    save(ACTIVITY_STORAGE_KEY, defaultActivities);
  }
};

ensureSeedData();

export const projectRepository = {
  list(): ProjectRecord[] {
    return load(PROJECT_STORAGE_KEY, defaultProjects);
  },
  get(projectId: string): ProjectRecord | undefined {
    return this.list().find((project) => project.id === projectId);
  },
  create(payload: Omit<ProjectRecord, 'id' | 'createdAt' | 'updatedAt'>): ProjectRecord {
    const projects = this.list();
    const newProject: ProjectRecord = {
      ...payload,
      id: `proj-${Date.now()}`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    save(PROJECT_STORAGE_KEY, [newProject, ...projects]);
    return newProject;
  },
  update(projectId: string, updates: Partial<ProjectRecord>): ProjectRecord | undefined {
    const projects = this.list();
    const idx = projects.findIndex((project) => project.id === projectId);
    if (idx === -1) return undefined;
    const updated: ProjectRecord = {
      ...projects[idx],
      ...updates,
      updatedAt: new Date().toISOString(),
    };
    const next = [...projects];
    next[idx] = updated;
    save(PROJECT_STORAGE_KEY, next);
    return updated;
  },
  remove(projectId: string): void {
    const remaining = this.list().filter((project) => project.id !== projectId);
    save(PROJECT_STORAGE_KEY, remaining);
    const activities = this.listActivities().filter(
      (activity) => activity.projectId !== projectId
    );
    save(ACTIVITY_STORAGE_KEY, activities);
  },
  listActivities(projectId?: string): ProjectActivity[] {
    const activities = load(ACTIVITY_STORAGE_KEY, defaultActivities);
    return projectId
      ? activities.filter((activity) => activity.projectId === projectId)
      : activities;
  },
  addActivity(activity: ProjectActivity): void {
    const activities = this.listActivities();
    save(ACTIVITY_STORAGE_KEY, [activity, ...activities]);
  },
  reset(): void {
    save(PROJECT_STORAGE_KEY, defaultProjects);
    save(ACTIVITY_STORAGE_KEY, defaultActivities);
  },
};

export default projectRepository;

