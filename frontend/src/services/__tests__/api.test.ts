/**
 * API服务单元测试
 */
import axios from 'axios';
import ApiService, { apiService } from '../api';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
});

// Mock window.location
delete (window as any).location;
window.location = { href: '' } as any;

describe('ApiService测试', () => {
  let apiServiceInstance: ApiService;
  let mockAxiosInstance: any;

  beforeEach(() => {
    // 重置所有mock
    jest.clearAllMocks();
    
    // 创建mock axios实例
    mockAxiosInstance = {
      get: jest.fn(),
      post: jest.fn(),
      delete: jest.fn(),
      interceptors: {
        request: { use: jest.fn() },
        response: { use: jest.fn() }
      }
    };
    
    mockedAxios.create.mockReturnValue(mockAxiosInstance);
    
    // 创建新的ApiService实例
    apiServiceInstance = new ApiService();
  });

  describe('构造函数测试', () => {
    test('应该使用默认baseURL创建axios实例', () => {
      new ApiService();
      
      expect(mockedAxios.create).toHaveBeenCalledWith({
        baseURL: 'http://localhost:8000',
        timeout: 30000,
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    test('应该使用自定义baseURL创建axios实例', () => {
      const customURL = 'https://api.example.com';
      new ApiService(customURL);
      
      expect(mockedAxios.create).toHaveBeenCalledWith({
        baseURL: customURL,
        timeout: 30000,
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    test('应该设置请求和响应拦截器', () => {
      new ApiService();
      
      expect(mockAxiosInstance.interceptors.request.use).toHaveBeenCalled();
      expect(mockAxiosInstance.interceptors.response.use).toHaveBeenCalled();
    });
  });

  describe('healthCheck方法测试', () => {
    test('应该成功返回健康检查结果', async () => {
      const mockResponse = { data: { status: 'ok' } };
      mockAxiosInstance.get.mockResolvedValue(mockResponse);

      const result = await apiServiceInstance.healthCheck();

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/health');
      expect(result).toEqual({
        success: true,
        data: mockResponse.data
      });
    });

    test('应该处理健康检查失败', async () => {
      const mockError = new Error('Network error');
      mockAxiosInstance.get.mockRejectedValue(mockError);

      const result = await apiServiceInstance.healthCheck();

      expect(result).toEqual({
        success: false,
        error: 'Network error'
      });
    });
  });

  describe('analyzeCode方法测试', () => {
    test('应该成功分析代码', async () => {
      const mockRequest = {
        code: 'console.log("hello");',
        language: 'javascript',
        analysis_type: 'quality' as const
      };
      const mockResponse = {
        data: {
          analysis_id: '123',
          quality_score: 85,
          issues: [],
          metrics: { complexity: 1, maintainability: 90, testability: 80 }
        }
      };
      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      const result = await apiServiceInstance.analyzeCode(mockRequest);

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/api/v1/ai/analyze', mockRequest);
      expect(result).toEqual({
        success: true,
        data: mockResponse.data
      });
    });

    test('应该处理代码分析失败', async () => {
      const mockRequest = {
        code: 'invalid code',
        language: 'javascript',
        analysis_type: 'quality' as const
      };
      const mockError = new Error('Analysis failed');
      mockAxiosInstance.post.mockRejectedValue(mockError);

      const result = await apiServiceInstance.analyzeCode(mockRequest);

      expect(result).toEqual({
        success: false,
        error: 'Analysis failed'
      });
    });
  });

  describe('getProjects方法测试', () => {
    test('应该成功获取项目列表', async () => {
      const mockProjects = [
        { id: '1', name: 'Project 1', description: 'Desc 1', created_at: '2023-01-01', updated_at: '2023-01-01' },
        { id: '2', name: 'Project 2', description: 'Desc 2', created_at: '2023-01-02', updated_at: '2023-01-02' }
      ];
      mockAxiosInstance.get.mockResolvedValue({ data: mockProjects });

      const result = await apiServiceInstance.getProjects();

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/api/v1/projects');
      expect(result).toEqual({
        success: true,
        data: mockProjects
      });
    });
  });

  describe('createProject方法测试', () => {
    test('应该成功创建项目', async () => {
      const mockProject = { name: 'New Project', description: 'New Description' };
      const mockResponse = {
        data: { 
          id: '123', 
          ...mockProject, 
          created_at: '2023-01-01', 
          updated_at: '2023-01-01' 
        }
      };
      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      const result = await apiServiceInstance.createProject(mockProject);

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/api/v1/projects', mockProject);
      expect(result).toEqual({
        success: true,
        data: mockResponse.data
      });
    });
  });

  describe('deleteProject方法测试', () => {
    test('应该成功删除项目', async () => {
      const projectId = '123';
      mockAxiosInstance.delete.mockResolvedValue({});

      const result = await apiServiceInstance.deleteProject(projectId);

      expect(mockAxiosInstance.delete).toHaveBeenCalledWith('/api/v1/projects/123');
      expect(result).toEqual({ success: true });
    });
  });

  describe('embedCode方法测试', () => {
    test('应该成功获取代码嵌入向量', async () => {
      const code = 'function test() {}';
      const mockResponse = { data: { embedding: [0.1, 0.2, 0.3] } };
      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      const result = await apiServiceInstance.embedCode(code);

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/api/v1/ai/embed', { code });
      expect(result).toEqual({
        success: true,
        data: [0.1, 0.2, 0.3]
      });
    });
  });

  describe('detectDefects方法测试', () => {
    test('应该成功检测代码缺陷', async () => {
      const code = 'function test() {}';
      const mockResponse = { data: { defects: [] } };
      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      const result = await apiServiceInstance.detectDefects(code);

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/api/v1/ai/defect-analysis', { code });
      expect(result).toEqual({
        success: true,
        data: mockResponse.data
      });
    });
  });

  describe('单例实例测试', () => {
    test('apiService应该是ApiService的实例', () => {
      expect(apiService).toBeInstanceOf(ApiService);
    });
  });
});