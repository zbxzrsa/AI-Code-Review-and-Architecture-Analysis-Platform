// Personalized Recommendation Engine
// Provides intelligent recommendations based on user profiles and behavioral data

// Local type definitions replacing removed internationalization modules
export type RegionCode = 'CN' | 'US' | 'JP' | 'DE' | 'FR';
export enum TechnicalLevelType { BEGINNER='BEGINNER', INTERMEDIATE='INTERMEDIATE', ADVANCED='ADVANCED', EXPERT='EXPERT' }
export enum LearningStyleType { VISUAL='VISUAL', READING_WRITING='READING_WRITING', KINESTHETIC='KINESTHETIC' }
export enum WorkContextType { PERSONAL='PERSONAL', ACADEMIC='ACADEMIC', CORPORATE='CORPORATE', STARTUP='STARTUP' }
export type UserTraitType = Record<string, number>;
export interface UserProfile {
  userId: string;
  region: RegionCode | string;
  primaryLanguage: string;
  technicalLevel: TechnicalLevelType;
  learningStyles: LearningStyleType[];
  workContext: WorkContextType;
  lastActive: number;
  featureUsageHistory: Record<string, number>;
  contentInteractions: Record<string, { interactionCount: number; lastInteraction: number; feedback: FeedbackType }>;
  learningProgress: Record<string, { progress: number; completed: boolean; completionDate: number | null }>;
}
export interface ProfileAnalysisResult { userId: string; traits: Record<string, number>; updatedAt: number }
export enum BehaviorDimension { FEATURE_USAGE_FREQUENCY='FEATURE_USAGE_FREQUENCY', FEATURE_USAGE_DEPTH='FEATURE_USAGE_DEPTH', COLLABORATION_LEVEL='COLLABORATION_LEVEL', LEARNING_SPEED='LEARNING_SPEED' }
export const userProfileSystem = {
  profiles: new Map<string, UserProfile>(),
  trackBehavior(_userId: string, _dimension: BehaviorDimension, _delta: number) {},
  getProfile(userId: string): UserProfile | undefined { return this.profiles.get(userId); },
  updateProfile(update: Partial<UserProfile> & { userId: string }) {
    const current = this.profiles.get(update.userId);
    const merged = { ...(current || {}), ...update } as UserProfile;
    this.profiles.set(update.userId, merged);
  },
  getLatestProfileAnalysis(userId: string): ProfileAnalysisResult | undefined { return { userId, traits: {}, updatedAt: Date.now() }; },
  generateProfileAnalysis(userId: string): ProfileAnalysisResult { return { userId, traits: {}, updatedAt: Date.now() }; },
  getUserSegments(_userId: string): Array<{ id: string }> { return []; },
};

// 推荐项目类型
export enum RecommendationItemType {
  FEATURE = 'feature',
  CONTENT = 'content',
  INTERFACE = 'interface',
  LEARNING_PATH = 'learning_path'
}

// 推荐项目
export interface RecommendationItem {
  id: string;
  type: RecommendationItemType;
  title: string;
  description: string;
  relevanceScore: number;
  tags: string[];
  regions: RegionCode[];
  technicalLevels: TechnicalLevelType[];
  learningStyles?: LearningStyleType[];
  workContexts?: WorkContextType[];
  imageUrl?: string;
  actionUrl?: string;
  createdAt: number;
}

// 推荐结果
export interface RecommendationResult {
  userId: string;
  timestamp: number;
  items: RecommendationItem[];
  context: {
    location: string;
    currentActivity: string;
    timeOfDay: string;
    deviceType: string;
  };
  metrics: {
    diversity: number;
    novelty: number;
    relevance: number;
  };
}

// 用户反馈类型
export enum FeedbackType {
  CLICK = 'click',
  DISMISS = 'dismiss',
  LIKE = 'like',
  DISLIKE = 'dislike',
  SAVE = 'save',
  SHARE = 'share',
  COMPLETE = 'complete'
}

// 用户反馈
export interface UserFeedback {
  userId: string;
  itemId: string;
  feedbackType: FeedbackType;
  timestamp: number;
  context?: Record<string, any>;
}

// 个性化推荐引擎
export class RecommendationEngine {
  private static instance: RecommendationEngine;
  private items: RecommendationItem[] = [];
  private userFeedback: Map<string, UserFeedback[]> = new Map();
  private recommendationHistory: Map<string, RecommendationResult[]> = new Map();
  
  private constructor() {
    this.initializeRecommendationItems();
  }

  public static getInstance(): RecommendationEngine {
    if (!RecommendationEngine.instance) {
      RecommendationEngine.instance = new RecommendationEngine();
    }
    return RecommendationEngine.instance;
  }

  // 初始化推荐项目
  private initializeRecommendationItems(): void {
    // 功能推荐项目
    this.items = [
      // 功能推荐 - 初学者
      {
        id: 'feature_code_templates',
        type: RecommendationItemType.FEATURE,
        title: 'Code Template Library',
        description: 'Start quickly with preset templates, ideal for beginners.',
        relevanceScore: 0.9,
        tags: ['beginner', 'productivity', 'templates'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.BEGINNER],
        workContexts: [WorkContextType.PERSONAL, WorkContextType.ACADEMIC],
        imageUrl: '/assets/features/code_templates.svg',
        actionUrl: '/features/code-templates',
        createdAt: Date.now()
      },
      {
        id: 'feature_guided_tutorials',
        type: RecommendationItemType.FEATURE,
        title: 'Guided Tutorials',
        description: 'Interactive learning experience with step-by-step guidance.',
        relevanceScore: 0.85,
        tags: ['beginner', 'learning', 'interactive'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.BEGINNER],
        learningStyles: [LearningStyleType.VISUAL, LearningStyleType.KINESTHETIC],
        imageUrl: '/assets/features/guided_tutorials.svg',
        actionUrl: '/features/guided-tutorials',
        createdAt: Date.now()
      },
      
      // 功能推荐 - 中级用户
      {
        id: 'feature_code_review',
        type: RecommendationItemType.FEATURE,
        title: 'Code Review Assistant',
        description: 'Analyze code quality and provide improvement suggestions intelligently.',
        relevanceScore: 0.8,
        tags: ['intermediate', 'code_quality', 'productivity'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.INTERMEDIATE],
        workContexts: [WorkContextType.CORPORATE, WorkContextType.STARTUP],
        imageUrl: '/assets/features/code_review.svg',
        actionUrl: '/features/code-review',
        createdAt: Date.now()
      },
      {
        id: 'feature_performance_analyzer',
        type: RecommendationItemType.FEATURE,
        title: 'Performance Analyzer',
        description: 'Identify performance bottlenecks and suggest optimizations.',
        relevanceScore: 0.75,
        tags: ['intermediate', 'performance', 'optimization'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.INTERMEDIATE, TechnicalLevelType.ADVANCED],
        workContexts: [WorkContextType.CORPORATE, WorkContextType.STARTUP],
        imageUrl: '/assets/features/performance_analyzer.svg',
        actionUrl: '/features/performance-analyzer',
        createdAt: Date.now()
      },
      
      // 功能推荐 - 高级用户
      {
        id: 'feature_architecture_visualization',
        type: RecommendationItemType.FEATURE,
        title: 'Architecture Visualization',
        description: 'Visualize code architecture and dependencies.',
        relevanceScore: 0.85,
        tags: ['advanced', 'architecture', 'visualization'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.ADVANCED, TechnicalLevelType.EXPERT],
        learningStyles: [LearningStyleType.VISUAL],
        workContexts: [WorkContextType.CORPORATE, WorkContextType.STARTUP],
        imageUrl: '/assets/features/architecture_visualization.svg',
        actionUrl: '/features/architecture-visualization',
        createdAt: Date.now()
      },
      {
        id: 'feature_advanced_refactoring',
        type: RecommendationItemType.FEATURE,
        title: 'Advanced Refactoring Tools',
        description: 'Intelligent code refactoring and pattern detection.',
        relevanceScore: 0.8,
        tags: ['advanced', 'refactoring', 'code_quality'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.ADVANCED, TechnicalLevelType.EXPERT],
        workContexts: [WorkContextType.CORPORATE, WorkContextType.STARTUP],
        imageUrl: '/assets/features/advanced_refactoring.svg',
        actionUrl: '/features/advanced-refactoring',
        createdAt: Date.now()
      },
      
      // 区域特定功能 - 中国
      {
        id: 'feature_cn_documentation',
        type: RecommendationItemType.FEATURE,
        title: 'Chinese Documentation and Tutorials',
        description: 'Comprehensive Chinese docs and localized tutorials.',
        relevanceScore: 0.9,
        tags: ['localization', 'documentation', 'chinese'],
        regions: ['CN'],
        technicalLevels: [TechnicalLevelType.BEGINNER, TechnicalLevelType.INTERMEDIATE, TechnicalLevelType.ADVANCED],
        imageUrl: '/assets/features/cn_documentation.svg',
        actionUrl: '/features/cn-documentation',
        createdAt: Date.now()
      },
      {
        id: 'feature_cn_api_integration',
        type: RecommendationItemType.FEATURE,
        title: 'China Local API Integration',
        description: 'Integrate Baidu Maps, Alipay, WeChat and other local services.',
        relevanceScore: 0.85,
        tags: ['localization', 'api', 'integration', 'chinese'],
        regions: ['CN'],
        technicalLevels: [TechnicalLevelType.INTERMEDIATE, TechnicalLevelType.ADVANCED],
        workContexts: [WorkContextType.CORPORATE, WorkContextType.STARTUP],
        imageUrl: '/assets/features/cn_api_integration.svg',
        actionUrl: '/features/cn-api-integration',
        createdAt: Date.now()
      },
      
      // 区域特定功能 - 日本
      {
        id: 'feature_jp_documentation',
        type: RecommendationItemType.FEATURE,
        title: 'Japanese Documentation and Tutorials',
        description: 'Comprehensive Japanese docs and localized tutorials.',
        relevanceScore: 0.9,
        tags: ['localization', 'documentation', 'japanese'],
        regions: ['JP'],
        technicalLevels: [TechnicalLevelType.BEGINNER, TechnicalLevelType.INTERMEDIATE, TechnicalLevelType.ADVANCED],
        imageUrl: '/assets/features/jp_documentation.svg',
        actionUrl: '/features/jp-documentation',
        createdAt: Date.now()
      },
      {
        id: 'feature_kanji_code_comments',
        type: RecommendationItemType.FEATURE,
        title: 'Kanji Code Comment Support',
        description: 'Proper display and handling of Kanji in code comments.',
        relevanceScore: 0.8,
        tags: ['localization', 'code_comments', 'japanese'],
        regions: ['JP'],
        technicalLevels: [TechnicalLevelType.BEGINNER, TechnicalLevelType.INTERMEDIATE, TechnicalLevelType.ADVANCED],
        imageUrl: '/assets/features/kanji_code_comments.svg',
        actionUrl: '/features/kanji-code-comments',
        createdAt: Date.now()
      },
      
      // 内容推荐
      {
        id: 'content_getting_started',
        type: RecommendationItemType.CONTENT,
        title: 'Getting Started Guide',
        description: 'Learn the basics of using the platform from scratch.',
        relevanceScore: 0.95,
        tags: ['beginner', 'tutorial', 'basics'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.BEGINNER],
        learningStyles: [LearningStyleType.VISUAL, LearningStyleType.READING_WRITING],
        imageUrl: '/assets/content/getting_started.svg',
        actionUrl: '/content/getting-started',
        createdAt: Date.now()
      },
      {
        id: 'content_best_practices',
        type: RecommendationItemType.CONTENT,
        title: 'Best Practices Guide',
        description: 'Best practices to improve code quality and development efficiency.',
        relevanceScore: 0.85,
        tags: ['intermediate', 'best_practices', 'code_quality'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.INTERMEDIATE, TechnicalLevelType.ADVANCED],
        learningStyles: [LearningStyleType.READING_WRITING],
        imageUrl: '/assets/content/best_practices.svg',
        actionUrl: '/content/best-practices',
        createdAt: Date.now()
      },
      {
        id: 'content_advanced_techniques',
        type: RecommendationItemType.CONTENT,
        title: 'Advanced Techniques Guide',
        description: 'Deep dive into advanced development techniques and architecture patterns.',
        relevanceScore: 0.8,
        tags: ['advanced', 'techniques', 'architecture'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.ADVANCED, TechnicalLevelType.EXPERT],
        learningStyles: [LearningStyleType.READING_WRITING],
        imageUrl: '/assets/content/advanced_techniques.svg',
        actionUrl: '/content/advanced-techniques',
        createdAt: Date.now()
      },
      
      // 区域特定内容 - 中国
      {
        id: 'content_cn_case_studies',
        type: RecommendationItemType.CONTENT,
        title: 'China Enterprise Case Studies',
        description: 'Real-world applications and best practices from Chinese tech companies.',
        relevanceScore: 0.9,
        tags: ['case_study', 'chinese', 'enterprise'],
        regions: ['CN'],
        technicalLevels: [TechnicalLevelType.INTERMEDIATE, TechnicalLevelType.ADVANCED],
        workContexts: [WorkContextType.CORPORATE, WorkContextType.STARTUP],
        imageUrl: '/assets/content/cn_case_studies.svg',
        actionUrl: '/content/cn-case-studies',
        createdAt: Date.now()
      },
      
      // 学习路径
      {
        id: 'path_beginner_to_intermediate',
        type: RecommendationItemType.LEARNING_PATH,
        title: 'Beginner to Intermediate Developer',
        description: 'Systematic learning path to grow from beginner to intermediate.',
        relevanceScore: 0.9,
        tags: ['learning_path', 'beginner', 'progression'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.BEGINNER],
        learningStyles: [LearningStyleType.VISUAL, LearningStyleType.READING_WRITING, LearningStyleType.KINESTHETIC],
        imageUrl: '/assets/paths/beginner_to_intermediate.svg',
        actionUrl: '/paths/beginner-to-intermediate',
        createdAt: Date.now()
      },
      {
        id: 'path_intermediate_to_advanced',
        type: RecommendationItemType.LEARNING_PATH,
        title: 'Intermediate to Advanced Developer',
        description: 'Study advanced development skills and architecture knowledge.',
        relevanceScore: 0.85,
        tags: ['learning_path', 'intermediate', 'advanced', 'progression'],
        regions: ['CN', 'US', 'JP', 'DE', 'FR'],
        technicalLevels: [TechnicalLevelType.INTERMEDIATE],
        learningStyles: [LearningStyleType.VISUAL, LearningStyleType.READING_WRITING, LearningStyleType.KINESTHETIC],
        imageUrl: '/assets/paths/intermediate_to_advanced.svg',
        actionUrl: '/paths/intermediate-to-advanced',
        createdAt: Date.now()
      }
    ];
  }

  // 获取推荐项目
  public getRecommendationItems(): RecommendationItem[] {
    return this.items;
  }

  // 添加推荐项目
  public addRecommendationItem(item: Omit<RecommendationItem, 'createdAt'>): RecommendationItem {
    const newItem: RecommendationItem = {
      ...item,
      createdAt: Date.now()
    };
    
    this.items.push(newItem);
    return newItem;
  }

  // 更新推荐项目
  public updateRecommendationItem(itemId: string, updates: Partial<Omit<RecommendationItem, 'id' | 'createdAt'>>): RecommendationItem | undefined {
    const index = this.items.findIndex(item => item.id === itemId);
    if (index === -1) {
      return undefined;
    }
    
    const updatedItem: RecommendationItem = {
      ...this.items[index],
      ...updates
    };
    
    this.items[index] = updatedItem;
    return updatedItem;
  }

  // 删除推荐项目
  public deleteRecommendationItem(itemId: string): boolean {
    const index = this.items.findIndex(item => item.id === itemId);
    if (index === -1) {
      return false;
    }
    
    this.items.splice(index, 1);
    return true;
  }

  // 记录用户反馈
  public recordFeedback(feedback: Omit<UserFeedback, 'timestamp'>): UserFeedback {
    const newFeedback: UserFeedback = {
      ...feedback,
      timestamp: Date.now()
    };
    
    if (!this.userFeedback.has(feedback.userId)) {
      this.userFeedback.set(feedback.userId, []);
    }
    
    const userFeedbacks = this.userFeedback.get(feedback.userId)!;
    userFeedbacks.push(newFeedback);
    
    // 更新用户行为数据
    this.updateUserBehaviorBasedOnFeedback(newFeedback);
    
    return newFeedback;
  }

  // 基于反馈更新用户行为数据
  private updateUserBehaviorBasedOnFeedback(feedback: UserFeedback): void {
    const { userId, feedbackType, itemId } = feedback;
    
    // 查找相关推荐项目
    const item = this.items.find(item => item.id === itemId);
    if (!item) {
      return;
    }
    
    // 根据反馈类型更新不同的行为维度
    switch (feedbackType) {
      case FeedbackType.CLICK:
        userProfileSystem.trackBehavior(userId, BehaviorDimension.FEATURE_USAGE_FREQUENCY, 1);
        break;
      case FeedbackType.LIKE:
        userProfileSystem.trackBehavior(userId, BehaviorDimension.FEATURE_USAGE_DEPTH, 2);
        break;
      case FeedbackType.DISLIKE:
        userProfileSystem.trackBehavior(userId, BehaviorDimension.FEATURE_USAGE_DEPTH, -1);
        break;
      case FeedbackType.SAVE:
        userProfileSystem.trackBehavior(userId, BehaviorDimension.FEATURE_USAGE_DEPTH, 3);
        break;
      case FeedbackType.SHARE:
        userProfileSystem.trackBehavior(userId, BehaviorDimension.COLLABORATION_LEVEL, 2);
        break;
      case FeedbackType.COMPLETE:
        userProfileSystem.trackBehavior(userId, BehaviorDimension.LEARNING_SPEED, 2);
        break;
    }
    
    // 更新用户画像中的特定交互记录
    const profile = userProfileSystem.getProfile(userId);
    if (profile) {
      // 更新功能使用历史
      if (item.type === RecommendationItemType.FEATURE) {
        const featureUsage = { ...profile.featureUsageHistory };
        featureUsage[item.id] = (featureUsage[item.id] || 0) + 1;
        
        userProfileSystem.updateProfile({
          userId,
          featureUsageHistory: featureUsage
        });
      }
      
      // 更新内容交互历史
      if (item.type === RecommendationItemType.CONTENT) {
        const contentInteractions = { ...profile.contentInteractions };
        contentInteractions[item.id] = {
          interactionCount: (contentInteractions[item.id]?.interactionCount || 0) + 1,
          lastInteraction: Date.now(),
          feedback: feedbackType
        };
        
        userProfileSystem.updateProfile({
          userId,
          contentInteractions
        });
      }
      
      // 更新学习路径进度
      if (item.type === RecommendationItemType.LEARNING_PATH && feedbackType === FeedbackType.COMPLETE) {
        const learningProgress = { ...profile.learningProgress };
        learningProgress[item.id] = {
          completed: true,
          completionDate: Date.now(),
          progress: 1
        };
        
        userProfileSystem.updateProfile({
          userId,
          learningProgress
        });
      }
    }
  }

  // 获取用户反馈
  public getUserFeedback(userId: string): UserFeedback[] {
    return this.userFeedback.get(userId) || [];
  }

  // 生成个性化推荐
  public generateRecommendations(userId: string, context: RecommendationResult['context'], count: number = 5): RecommendationResult {
    // 获取用户画像
    const profile = userProfileSystem.getProfile(userId);
    if (!profile) {
      // 如果没有用户画像，返回通用推荐
      return this.generateGenericRecommendations(userId, context, count);
    }
    
    // 获取用户画像分析结果
    const profileAnalysis = userProfileSystem.getLatestProfileAnalysis(userId) || userProfileSystem.generateProfileAnalysis(userId);
    if (!profileAnalysis) {
      // 如果无法生成分析结果，返回通用推荐
      return this.generateGenericRecommendations(userId, context, count);
    }
    
    // 获取用户反馈历史
    const feedbackHistory = this.getUserFeedback(userId);
    
    // 已经推荐过的项目ID
    const recommendedItemIds = new Set<string>();
    this.recommendationHistory.get(userId)?.forEach(result => {
      result.items.forEach(item => recommendedItemIds.add(item.id));
    });
    
    // 已经有负面反馈的项目ID
    const dislikedItemIds = new Set<string>();
    feedbackHistory
      .filter(feedback => feedback.feedbackType === FeedbackType.DISLIKE || feedback.feedbackType === FeedbackType.DISMISS)
      .forEach(feedback => dislikedItemIds.add(feedback.itemId));
    
    // 计算每个推荐项目的相关性得分
    const scoredItems = this.items
      .filter(item => !dislikedItemIds.has(item.id)) // 过滤掉不喜欢的项目
      .map(item => {
        // 基础相关性得分
        let score = item.relevanceScore;
        
        // 根据用户特征调整得分
        
        // 技术水平匹配度
        if (item.technicalLevels.includes(profile.technicalLevel)) {
          score += 0.2;
        } else {
          score -= 0.3;
        }
        
        // 区域匹配度
        if (item.regions.includes(profile.region as RegionCode)) {
          score += 0.15;
        }
        
        // 学习风格匹配度
        if (item.learningStyles && profile.learningStyles.some((style: LearningStyleType) => item.learningStyles?.includes(style))) {
          score += 0.1;
        }
        
        // 工作环境匹配度
        if (item.workContexts && item.workContexts.includes(profile.workContext)) {
          score += 0.1;
        }
        
        // 根据用户行为调整得分
        
        // 如果用户之前与类似项目有积极互动，提高得分
        const similarItemFeedback = feedbackHistory.filter(feedback => {
          const feedbackItem = this.items.find(i => i.id === feedback.itemId);
          return feedbackItem && 
                 feedbackItem.type === item.type && 
                 feedbackItem.tags.some(tag => item.tags.includes(tag));
        });
        
        const positiveFeedbackCount = similarItemFeedback.filter(feedback => 
          [FeedbackType.LIKE, FeedbackType.SAVE, FeedbackType.SHARE, FeedbackType.COMPLETE].includes(feedback.feedbackType)
        ).length;
        
        score += positiveFeedbackCount * 0.05;
        
        // 如果是新项目，稍微提高得分以促进探索
        if (!recommendedItemIds.has(item.id)) {
          score += 0.05;
        }
        
        // 根据上下文调整得分
        
        // 时间相关性（例如，晚上推荐深色主题）
        const hour = new Date().getHours();
        if (context.timeOfDay === 'evening' && item.tags.includes('dark_theme')) {
          score += 0.1;
        }
        
        // 设备相关性
        if (context.deviceType === 'mobile' && item.tags.includes('mobile_friendly')) {
          score += 0.1;
        }
        
        // 活动相关性
        if (context.currentActivity === 'coding' && 
            (item.type === RecommendationItemType.FEATURE || item.tags.includes('productivity'))) {
          score += 0.1;
        }
        
        return {
          item,
          score
        };
      })
      .sort((a, b) => b.score - a.score); // 按得分降序排序
    
    // 选择前N个项目
    const selectedItems = scoredItems.slice(0, count).map(({ item }) => item);
    
    // 计算推荐指标
    const metrics = this.calculateRecommendationMetrics(selectedItems, feedbackHistory);
    
    // 创建推荐结果
    const result: RecommendationResult = {
      userId,
      timestamp: Date.now(),
      items: selectedItems,
      context,
      metrics
    };
    
    // 保存推荐历史
    if (!this.recommendationHistory.has(userId)) {
      this.recommendationHistory.set(userId, []);
    }
    
    const userHistory = this.recommendationHistory.get(userId)!;
    userHistory.push(result);
    
    // 限制历史记录数量
    if (userHistory.length > 20) {
      this.recommendationHistory.set(userId, userHistory.slice(-20));
    }
    
    return result;
  }

  // 生成通用推荐（无用户画像时）
  private generateGenericRecommendations(userId: string, context: RecommendationResult['context'], count: number): RecommendationResult {
    // 选择评分较高的通用项目
    const selectedItems = this.items
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, count);
    
    // 计算推荐指标
    const metrics = {
      diversity: 0.5, // 中等多样性
      novelty: 0.7,   // 较高新颖性（因为用户是新用户）
      relevance: 0.5  // 中等相关性（因为没有用户画像）
    };
    
    // 创建推荐结果
    const result: RecommendationResult = {
      userId,
      timestamp: Date.now(),
      items: selectedItems,
      context,
      metrics
    };
    
    // 保存推荐历史
    if (!this.recommendationHistory.has(userId)) {
      this.recommendationHistory.set(userId, []);
    }
    
    const userHistory = this.recommendationHistory.get(userId)!;
    userHistory.push(result);
    
    return result;
  }

  // 计算推荐指标
  private calculateRecommendationMetrics(items: RecommendationItem[], feedbackHistory: UserFeedback[]): RecommendationResult['metrics'] {
    // 多样性：不同类型和标签的项目比例
    const typeCount = new Set(items.map(item => item.type)).size;
    const tagSet = new Set<string>();
    items.forEach(item => item.tags.forEach(tag => tagSet.add(tag)));
    const diversity = (typeCount / Object.keys(RecommendationItemType).length + tagSet.size / 20) / 2; // 归一化到0-1
    
    // 新颖性：之前未推荐过的项目比例
    const previouslyRecommendedIds = new Set<string>();
    feedbackHistory.forEach(feedback => previouslyRecommendedIds.add(feedback.itemId));
    const novelItems = items.filter(item => !previouslyRecommendedIds.has(item.id));
    const novelty = novelItems.length / items.length;
    
    // 相关性：平均相关性得分
    const relevance = items.reduce((sum, item) => sum + item.relevanceScore, 0) / items.length;
    
    return {
      diversity,
      novelty,
      relevance
    };
  }

  // 获取推荐历史
  public getRecommendationHistory(userId: string): RecommendationResult[] {
    return this.recommendationHistory.get(userId) || [];
  }

  // 评估推荐效果
  public evaluateRecommendationEffectiveness(userId: string, days: number = 30): Record<string, any> {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    // 获取指定时间范围内的推荐历史
    const history = this.getRecommendationHistory(userId).filter(
      result => result.timestamp >= startDate.getTime() && result.timestamp <= endDate.getTime()
    );
    
    if (history.length === 0) {
      return {
        userId,
        period: `${days} days`,
        error: 'Not enough recommendation history data'
      };
    }
    
    // 获取用户反馈
    const feedbackHistory = this.getUserFeedback(userId).filter(
      feedback => feedback.timestamp >= startDate.getTime() && feedback.timestamp <= endDate.getTime()
    );
    
    // 计算点击率 (CTR)
    const recommendedItemIds = new Set<string>();
    history.forEach(result => result.items.forEach(item => recommendedItemIds.add(item.id)));
    
    const clickedItemIds = new Set<string>();
    feedbackHistory
      .filter(feedback => feedback.feedbackType === FeedbackType.CLICK)
      .forEach(feedback => clickedItemIds.add(feedback.itemId));
    
    const ctr = recommendedItemIds.size > 0 ? clickedItemIds.size / recommendedItemIds.size : 0;
    
    // 计算转化率（完成/保存/分享）
    const convertedItemIds = new Set<string>();
    feedbackHistory
      .filter(feedback => [FeedbackType.COMPLETE, FeedbackType.SAVE, FeedbackType.SHARE].includes(feedback.feedbackType))
      .forEach(feedback => convertedItemIds.add(feedback.itemId));
    
    const conversionRate = clickedItemIds.size > 0 ? convertedItemIds.size / clickedItemIds.size : 0;
    
    // 计算满意度（喜欢/不喜欢比率）
    const likedItems = feedbackHistory.filter(feedback => feedback.feedbackType === FeedbackType.LIKE).length;
    const dislikedItems = feedbackHistory.filter(feedback => feedback.feedbackType === FeedbackType.DISLIKE).length;
    
    const satisfactionRate = (likedItems + dislikedItems) > 0 ? likedItems / (likedItems + dislikedItems) : 0;
    
    // 计算平均指标
    const avgDiversity = history.reduce((sum, result) => sum + result.metrics.diversity, 0) / history.length;
    const avgNovelty = history.reduce((sum, result) => sum + result.metrics.novelty, 0) / history.length;
    const avgRelevance = history.reduce((sum, result) => sum + result.metrics.relevance, 0) / history.length;
    
    // 按推荐类型分析
    const typeAnalysis: Record<RecommendationItemType, { count: number, clicks: number, conversions: number }> = {
      [RecommendationItemType.FEATURE]: { count: 0, clicks: 0, conversions: 0 },
      [RecommendationItemType.CONTENT]: { count: 0, clicks: 0, conversions: 0 },
      [RecommendationItemType.INTERFACE]: { count: 0, clicks: 0, conversions: 0 },
      [RecommendationItemType.LEARNING_PATH]: { count: 0, clicks: 0, conversions: 0 }
    };
    
    // 统计各类型的推荐数量
    history.forEach(result => {
      result.items.forEach(item => {
        typeAnalysis[item.type].count++;
      });
    });
    
    // 统计各类型的点击和转化
    feedbackHistory.forEach(feedback => {
      const item = this.items.find(item => item.id === feedback.itemId);
      if (!item) return;
      
      if (feedback.feedbackType === FeedbackType.CLICK) {
        typeAnalysis[item.type].clicks++;
      }
      
      if ([FeedbackType.COMPLETE, FeedbackType.SAVE, FeedbackType.SHARE].includes(feedback.feedbackType)) {
        typeAnalysis[item.type].conversions++;
      }
    });
    
    return {
      userId,
      period: `${days} days`,
      recommendationCount: history.length,
      uniqueItemsRecommended: recommendedItemIds.size,
      metrics: {
        ctr,
        conversionRate,
        satisfactionRate,
        avgDiversity,
        avgNovelty,
        avgRelevance
      },
      typeAnalysis: Object.entries(typeAnalysis).map(([type, data]) => ({
        type,
        count: data.count,
        clicks: data.clicks,
        conversions: data.conversions,
        ctr: data.count > 0 ? data.clicks / data.count : 0,
        conversionRate: data.clicks > 0 ? data.conversions / data.clicks : 0
      }))
    };
  }

  // A/B测试功能
  public createABTest(testConfig: {
    id: string;
    name: string;
    description: string;
    variantA: string[];
    variantB: string[];
    userSegmentIds: string[];
    startDate: number;
    endDate: number;
  }): void {
    // 这里应该实现A/B测试逻辑
    // 简化实现，仅记录测试配置
    console.log('Create A/B test:', testConfig);
  }

  // 获取个性化仪表板数据
  public getPersonalizationDashboardData(userId: string): Record<string, any> {
    // 获取用户画像
    const profile = userProfileSystem.getProfile(userId);
    if (!profile) {
      return { error: 'User profile not found' };
    }
    
    // 获取用户画像分析
    const profileAnalysis = userProfileSystem.getLatestProfileAnalysis(userId);
    
    // 获取推荐历史
    const recommendationHistory = this.getRecommendationHistory(userId);
    
    // 获取反馈历史
    const feedbackHistory = this.getUserFeedback(userId);
    
    // 计算推荐效果
    const effectiveness = this.evaluateRecommendationEffectiveness(userId);
    
    // 用户分群
    const segments = userProfileSystem.getUserSegments(userId).map((segment: { id: string }) => segment.id);
    
    // 学习进度
    // 学习进度
    const progressEntries = Object.entries(profile.learningProgress);
    const completedContents = progressEntries.filter(([_, progress]) => progress.completed).length;
    const learningProgress = progressEntries.map(([pathId, data]) => ({
      pathId,
      progress: data.progress,
      completed: data.completed,
      completionDate: data.completionDate
    }));
    
    // 功能使用情况
    const featureUsage = Object.entries(profile.featureUsageHistory)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([featureId, count]) => ({
        featureId,
        count
      }));
    
    return {
      userId,
      lastUpdated: Date.now(),
      profile: {
        region: profile.region,
        primaryLanguage: profile.primaryLanguage,
        technicalLevel: profile.technicalLevel,
        learningStyles: profile.learningStyles,
        workContext: profile.workContext,
        lastActive: profile.lastActive
      },
      segments,
      analytics: {
        recommendationCount: recommendationHistory.length,
        feedbackCount: feedbackHistory.length,
        effectiveness
      },
      learningProgress,
      topFeatures: featureUsage,
      traits: profileAnalysis?.traits || {}
    };
  }
}

// 导出单例实例
export const recommendationEngine = RecommendationEngine.getInstance();