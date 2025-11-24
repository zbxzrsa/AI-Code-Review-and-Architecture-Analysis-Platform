/**
 * API服务模块
 * 提供与后端API通信的功能
 */
import axios, { AxiosInstance, AxiosResponse } from 'axios';

// API响应接口
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// 代码分析请求接口
export interface CodeAnalysisRequest {
  code: string;
  language: string;
  analysis_type: 'quality' | 'security' | 'architecture';
}

// 代码分析响应接口
export interface CodeAnalysisResponse {
  analysis_id: string;
  quality_score: number;
  issues: Array<{
    type: string;
    severity: 'low' | 'medium' | 'high';
    message: string;
    line: number;
  }>;
  metrics: {
    complexity: number;
    maintainability: number;
    testability: number;
  };
}

// 项目接口
export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

// 会话接口
export interface AnalysisSession {
  id: string;
  project_id: string;
  label: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at?: string;
  completed_at?: string;
  summary?: string;
}

// 会话工件接口
export interface SessionArtifact {
  id: string;
  session_id: string;
  type: 'code' | 'result' | 'log';
  path: string;
  size: number;
  created_at: string;
}

// 文件版本接口
export interface FileVersion {
  id: string;
  project_id: string;
  file_path: string;
  sha256: string;
  created_at: string;
}

// 搜索结果接口
export interface SearchResult {
  id: string;
  type: 'project' | 'session' | 'file' | 'issue';
  title: string;
  description: string;
  score: number;
  data: any;
}

// 基线接口
export interface Baseline {
  id: string;
  project_id: string;
  name: string;
  description: string;
  config: Record<string, any>;
  created_at: string;
}

// 基线偏差接口
export interface BaselineDeviation {
  id: string;
  baseline_id: string;
  metric_name: string;
  deviation_value: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  detected_at: string;
}

export interface LoginPayload {
  username: string;
  password: string;
}

export interface LoginResult {
  access_token: string;
  token_type: string;
  username: string;
}

 

class ApiService {
  private client: AxiosInstance;

  constructor(baseURL: string = process.env.REACT_APP_API_URL || 'http://localhost:8000') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        // 添加认证token等
        const token = localStorage.getItem('auth_token');
        if (token) {
          (config.headers as any).Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        if (error.response?.status === 401) {
          // 处理认证失败
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // 健康检查
  async healthCheck(): Promise<ApiResponse> {
    try {
      const response = await this.client.get('/health');
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 代码分析
  async analyzeCode(request: CodeAnalysisRequest): Promise<ApiResponse<CodeAnalysisResponse>> {
    try {
      const response = await this.client.post('/api/v1/ai/analyze', request);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取项目列表
  async getProjects(): Promise<ApiResponse<Project[]>> {
    try {
      const response = await this.client.get('/api/v1/projects');
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 创建项目
  async createProject(project: Omit<Project, 'id' | 'created_at' | 'updated_at'>): Promise<ApiResponse<Project>> {
    try {
      const response = await this.client.post('/api/v1/projects', project);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 删除项目
  async deleteProject(projectId: string): Promise<ApiResponse> {
    try {
      await this.client.delete(`/api/v1/projects/${projectId}`);
      return { success: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 代码嵌入
  async embedCode(code: string): Promise<ApiResponse<number[]>> {
    try {
      const response = await this.client.post('/api/v1/ai/embed', { code });
      return { success: true, data: response.data.embedding };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 缺陷检测
  async detectDefects(code: string): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.post('/api/v1/ai/defect-analysis', { code });
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // === 会话管理 API ===
  
  // 获取会话列表
  async getSessions(projectId?: string): Promise<ApiResponse<AnalysisSession[]>> {
    try {
      const url = projectId ? `/api/v1/sessions?project_id=${projectId}` : '/api/v1/sessions';
      const response = await this.client.get(url);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取特定会话
  async getSession(sessionId: string): Promise<ApiResponse<AnalysisSession>> {
    try {
      const response = await this.client.get(`/api/v1/sessions/${sessionId}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 创建会话
  async createSession(session: {
    project_id: string;
    label: string;
    description?: string;
  }): Promise<ApiResponse<AnalysisSession>> {
    try {
      const response = await this.client.post('/api/v1/sessions', session);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 更新会话
  async updateSession(sessionId: string, updates: Partial<AnalysisSession>): Promise<ApiResponse<AnalysisSession>> {
    try {
      const response = await this.client.put(`/api/v1/sessions/${sessionId}`, updates);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 删除会话
  async deleteSession(sessionId: string): Promise<ApiResponse> {
    try {
      await this.client.delete(`/api/v1/sessions/${sessionId}`);
      return { success: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取会话工件
  async getSessionArtifacts(sessionId: string): Promise<ApiResponse<SessionArtifact[]>> {
    try {
      const response = await this.client.get(`/api/v1/sessions/${sessionId}/artifacts`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // === 版本管理 API ===

  // 获取文件版本列表
  async getFileVersions(projectId: string): Promise<ApiResponse<FileVersion[]>> {
    try {
      const response = await this.client.get(`/api/v1/versions?project_id=${projectId}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取特定文件版本
  async getFileVersion(versionId: string): Promise<ApiResponse<FileVersion>> {
    try {
      const response = await this.client.get(`/api/v1/versions/${versionId}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 创建文件版本
  async createFileVersion(version: {
    project_id: string;
    file_path: string;
    content: string;
  }): Promise<ApiResponse<FileVersion>> {
    try {
      const response = await this.client.post('/api/v1/versions', version);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取项目文件列表
  async getProjectFiles(projectId: string): Promise<ApiResponse<Array<{path: string, version_count: number}>>> {
    try {
      const response = await this.client.get(`/api/v1/versions/files?project_id=${projectId}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取文件历史
  async getFileHistory(projectId: string, filePath: string): Promise<ApiResponse<FileVersion[]>> {
    try {
      const response = await this.client.get(`/api/v1/versions/history?project_id=${projectId}&path=${encodeURIComponent(filePath)}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 版本比较
  async compareVersions(version1Id: string, version2Id: string): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.post('/api/v1/versions/compare', { version1_id: version1Id, version2_id: version2Id });
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // === 搜索 API ===

  // 全文搜索
  async search(query: string, filters?: {
    type?: string;
    project_id?: string;
    date_from?: string;
    date_to?: string;
  }): Promise<ApiResponse<SearchResult[]>> {
    try {
      const params = new URLSearchParams({ q: query });
      if (filters) {
        Object.entries(filters).forEach(([key, value]) => {
          if (value) params.append(key, value);
        });
      }
      const response = await this.client.get(`/api/v1/search?${params.toString()}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取搜索建议
  async getSearchSuggestions(query: string): Promise<ApiResponse<string[]>> {
    try {
      const response = await this.client.get(`/api/v1/search/suggestions?q=${encodeURIComponent(query)}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取可用过滤器
  async getSearchFilters(): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get('/api/v1/search/filters');
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  

  // === 基线管理 API ===

  // 获取基线列表
  async getBaselines(projectId?: string): Promise<ApiResponse<Baseline[]>> {
    try {
      const url = projectId ? `/api/v1/baselines?project_id=${projectId}` : '/api/v1/baselines';
      const response = await this.client.get(url);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取特定基线
  async getBaseline(baselineId: string): Promise<ApiResponse<Baseline>> {
    try {
      const response = await this.client.get(`/api/v1/baselines/${baselineId}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 创建基线
  async createBaseline(baseline: {
    project_id: string;
    name: string;
    description?: string;
    config: Record<string, any>;
  }): Promise<ApiResponse<Baseline>> {
    try {
      const response = await this.client.post('/api/v1/baselines', baseline);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 更新基线
  async updateBaseline(baselineId: string, updates: Partial<Baseline>): Promise<ApiResponse<Baseline>> {
    try {
      const response = await this.client.put(`/api/v1/baselines/${baselineId}`, updates);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 删除基线
  async deleteBaseline(baselineId: string): Promise<ApiResponse> {
    try {
      await this.client.delete(`/api/v1/baselines/${baselineId}`);
      return { success: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取基线偏差
  async getBaselineDeviations(baselineId: string): Promise<ApiResponse<BaselineDeviation[]>> {
    try {
      const response = await this.client.get(`/api/v1/baselines/${baselineId}/deviations`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 创建基线偏差
  async createBaselineDeviation(deviation: {
    baseline_id: string;
    metric_name: string;
    deviation_value: number;
    severity: 'low' | 'medium' | 'high' | 'critical';
  }): Promise<ApiResponse<BaselineDeviation>> {
    try {
      const response = await this.client.post('/api/v1/baselines/deviations', deviation);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // 获取基线状态
  async getBaselineStatus(baselineId: string): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get(`/api/v1/baselines/${baselineId}/status`);
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // === 认证 API ===
  async login(payload: LoginPayload): Promise<ApiResponse<LoginResult>> {
    try {
      const response = await this.client.post('/api/v1/auth/login', payload);
      const data = response.data as LoginResult;
      localStorage.setItem('auth_token', data.access_token);
      return { success: true, data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async currentUser(): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get('/api/v1/auth/me');
      return { success: true, data: response.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

// 导出单例实例
export const apiService = new ApiService();
export default ApiService;