import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Progress, Badge, Tooltip, notification, Button } from 'antd';
import { TrophyOutlined, RocketOutlined, StarOutlined, LockOutlined } from '@ant-design/icons';
import MicroInteractions from '../ui/MicroInteractions';

const { Title, Text, Paragraph } = Typography;
const { SuccessAnimation, FadeIn } = MicroInteractions;

// Achievement types
export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  category: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  progress: number;
  maxProgress: number;
  completed: boolean;
  unlocked: boolean;
  xpReward: number;
  dateCompleted?: string;
}

// User level definition
export interface UserLevel {
  level: number;
  title: string;
  minXP: number;
  maxXP: number;
  benefits: string[];
}

// Achievement system props
interface AchievementSystemProps {
  userId: string;
  onAchievementCompleted?: (achievement: Achievement) => void;
}

// User level config
const USER_LEVELS: UserLevel[] = [
  {
    level: 1,
    title: 'Junior Analyst',
    minXP: 0,
    maxXP: 100,
    benefits: ['Basic features', 'Daily tips']
  },
  {
    level: 2,
    title: 'Associate Analyst',
    minXP: 100,
    maxXP: 300,
    benefits: ['Advanced search', 'Custom dashboard']
  },
  {
    level: 3,
    title: 'Mid-level Analyst',
    minXP: 300,
    maxXP: 600,
    benefits: ['Batch analysis', 'Export advanced reports']
  },
  {
    level: 4,
    title: 'Senior Analyst',
    minXP: 600,
    maxXP: 1000,
    benefits: ['Team collaboration', 'Historical trend analysis']
  },
  {
    level: 5,
    title: 'Expert Analyst',
    minXP: 1000,
    maxXP: 2000,
    benefits: ['API access', 'Priority support']
  }
];

// Sample achievements
const SAMPLE_ACHIEVEMENTS: Achievement[] = [
  {
    id: 'first-analysis',
    title: 'First Code Analysis',
    description: 'Complete your first code analysis',
    icon: <RocketOutlined />,
    category: 'beginner',
    progress: 0,
    maxProgress: 1,
    completed: false,
    unlocked: true,
    xpReward: 10
  },
  {
    id: 'fix-issues',
    title: 'Issue Fixer',
    description: 'Fix 10 code issues',
    icon: <TrophyOutlined />,
    category: 'beginner',
    progress: 0,
    maxProgress: 10,
    completed: false,
    unlocked: true,
    xpReward: 20
  },
  {
    id: 'architecture-master',
    title: 'Architecture Master',
    description: 'Complete 5 architecture analyses',
    icon: <StarOutlined />,
    category: 'intermediate',
    progress: 0,
    maxProgress: 5,
    completed: false,
    unlocked: true,
    xpReward: 30
  },
  {
    id: 'code-quality-champion',
    title: 'Code Quality Champion',
    description: 'Raise project code quality above 90%',
    icon: <TrophyOutlined />,
    category: 'advanced',
    progress: 0,
    maxProgress: 1,
    completed: false,
    unlocked: false,
    xpReward: 50
  }
];

// Storage keys
const ACHIEVEMENTS_STORAGE_KEY = 'user_achievements';
const USER_XP_STORAGE_KEY = 'user_xp';

// Component
const AchievementSystem: React.FC<AchievementSystemProps> = ({ userId, onAchievementCompleted }) => {
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [userXP, setUserXP] = useState<number>(0);
  const [currentLevel, setCurrentLevel] = useState<UserLevel>(USER_LEVELS[0]);
  const [showNewAchievement, setShowNewAchievement] = useState<Achievement | null>(null);

  // Init achievements
  useEffect(() => {
    // Load from localStorage
    const storedAchievements = localStorage.getItem(`${ACHIEVEMENTS_STORAGE_KEY}_${userId}`);
    const storedXP = localStorage.getItem(`${USER_XP_STORAGE_KEY}_${userId}`);
    
    if (storedAchievements) {
      setAchievements(JSON.parse(storedAchievements));
    } else {
      setAchievements(SAMPLE_ACHIEVEMENTS);
    }
    
    if (storedXP) {
      setUserXP(parseInt(storedXP, 10));
    }
  }, [userId]);

  // Update user level
  useEffect(() => {
    const newLevel = USER_LEVELS.find(
      level => userXP >= level.minXP && userXP < level.maxXP
    ) || USER_LEVELS[USER_LEVELS.length - 1];
    
    setCurrentLevel(newLevel);
  }, [userXP]);

  // Persist achievements
  useEffect(() => {
    if (achievements.length > 0) {
      localStorage.setItem(`${ACHIEVEMENTS_STORAGE_KEY}_${userId}`, JSON.stringify(achievements));
    }
  }, [achievements, userId]);

  // Persist XP
  useEffect(() => {
    localStorage.setItem(`${USER_XP_STORAGE_KEY}_${userId}`, userXP.toString());
  }, [userXP, userId]);

  // Update achievement progress
  const updateAchievementProgress = (achievementId: string, progress: number) => {
    setAchievements(prevAchievements => {
      const updatedAchievements = prevAchievements.map(achievement => {
        if (achievement.id === achievementId) {
          const newProgress = Math.min(achievement.maxProgress, achievement.progress + progress);
          const wasCompleted = achievement.completed;
          const isNowCompleted = newProgress >= achievement.maxProgress;
          
          // Newly completed
          if (!wasCompleted && isNowCompleted) {
            // Add XP reward
            setUserXP(prev => prev + achievement.xpReward);
            
            // Show notification
            const completedAchievement = {
              ...achievement,
              progress: newProgress,
              completed: true,
              dateCompleted: new Date().toISOString()
            };
            
            setShowNewAchievement(completedAchievement);
            
            // Callback
            if (onAchievementCompleted) {
              onAchievementCompleted(completedAchievement);
            }
            
            return completedAchievement;
          }
          
          return {
            ...achievement,
            progress: newProgress,
            completed: isNowCompleted,
            dateCompleted: isNowCompleted && !achievement.dateCompleted 
              ? new Date().toISOString() 
              : achievement.dateCompleted
          };
        }
        return achievement;
      });
      
      return updatedAchievements;
    });
  };

  // Unlock achievement
  const unlockAchievement = (achievementId: string) => {
    setAchievements(prevAchievements => {
      return prevAchievements.map(achievement => {
        if (achievement.id === achievementId) {
          return {
            ...achievement,
            unlocked: true
          };
        }
        return achievement;
      });
    });
  };

  // Close notification
  const handleCloseAchievementNotification = () => {
    setShowNewAchievement(null);
  };

  // Notification
  useEffect(() => {
    if (showNewAchievement) {
      notification.open({
        message: 'Achievement Unlocked!',
        description: (
          <div>
            <SuccessAnimation show={true} />
            <Title level={5}>{showNewAchievement.title}</Title>
            <Paragraph>{showNewAchievement.description}</Paragraph>
            <Text type="success">+{showNewAchievement.xpReward} XP</Text>
          </div>
        ),
        duration: 5,
        placement: 'topRight',
        onClose: handleCloseAchievementNotification
      });
    }
  }, [showNewAchievement]);

  // XP to next level
  const xpToNextLevel = currentLevel.level < USER_LEVELS.length 
    ? USER_LEVELS[currentLevel.level].minXP - userXP 
    : 0;

  // Current level progress
  const levelProgressPercent = Math.floor(
    ((userXP - currentLevel.minXP) / (currentLevel.maxXP - currentLevel.minXP)) * 100
  );

  // Simulate progress (demo only)
  const simulateProgress = (achievementId: string) => {
    updateAchievementProgress(achievementId, 1);
  };

  return (
    <div className="achievement-system">
      <FadeIn>
        {/* User level card */}
        <Card className="user-level-card" style={{ marginBottom: 24 }}>
          <Row gutter={[16, 16]} align="middle">
            <Col span={6}>
              <div className="level-badge" style={{ textAlign: 'center' }}>
                <Badge count={currentLevel.level} overflowCount={10} style={{ backgroundColor: '#1677ff' }}>
                  <div style={{ 
                    width: 80, 
                    height: 80, 
                    borderRadius: '50%', 
                    background: '#f0f5ff', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    fontSize: 32
                  }}>
                    {currentLevel.level < 3 ? <RocketOutlined /> : 
                     currentLevel.level < 5 ? <TrophyOutlined /> : <StarOutlined />}
                  </div>
                </Badge>
                <Title level={4} style={{ marginTop: 16 }}>{currentLevel.title}</Title>
              </div>
            </Col>
            <Col span={18}>
              <div className="level-progress">
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Text>{userXP} XP</Text>
                  <Text>{currentLevel.maxXP} XP</Text>
                </div>
                <Progress percent={levelProgressPercent} />
                
                {currentLevel.level < USER_LEVELS.length && (
                  <Text type="secondary">
                    还需 {xpToNextLevel} XP 升级到 {USER_LEVELS[currentLevel.level].title}
                  </Text>
                )}
                
                <div style={{ marginTop: 16 }}>
                <Title level={5}>Current Level Benefits:</Title>
                  <ul>
                    {currentLevel.benefits.map((benefit, index) => (
                      <li key={index}>{benefit}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </Col>
          </Row>
        </Card>
        
        {/* Achievements */}
        <Title level={4}>Achievements</Title>
        <Row gutter={[16, 16]}>
          {achievements.map(achievement => (
            <Col xs={24} sm={12} md={8} key={achievement.id}>
              <Card 
                hoverable 
                className={`achievement-card ${achievement.completed ? 'completed' : ''} ${!achievement.unlocked ? 'locked' : ''}`}
              >
                <div style={{ opacity: achievement.unlocked ? 1 : 0.6 }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                    <div style={{ 
                      fontSize: 24, 
                      marginRight: 12,
                      color: achievement.completed ? '#52c41a' : '#1677ff'
                    }}>
                      {achievement.icon}
                    </div>
                    <div>
                      <Tooltip title={!achievement.unlocked ? 'Complete more tasks to unlock' : ''}>
                        <Title level={5} style={{ margin: 0 }}>
                          {achievement.title} {!achievement.unlocked && <LockOutlined />}
                        </Title>
                      </Tooltip>
                      <Text type="secondary">{achievement.description}</Text>
                    </div>
                  </div>
                  
                  <Progress 
                    percent={Math.floor((achievement.progress / achievement.maxProgress) * 100)} 
                    status={achievement.completed ? 'success' : 'active'}
                    size="small"
                  />
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
                    <Text>{achievement.progress}/{achievement.maxProgress}</Text>
                    <Text type="secondary">+{achievement.xpReward} XP</Text>
                  </div>
                  
                  {achievement.completed && achievement.dateCompleted && (
                    <div style={{ marginTop: 8 }}>
                      <Text type="success">Completed on: {new Date(achievement.dateCompleted).toLocaleDateString()}</Text>
                    </div>
                  )}
                  
                  {/* Demo button */}
                  {!achievement.completed && achievement.unlocked && (
                    <Button 
                      size="small" 
                      style={{ marginTop: 8 }} 
                      onClick={() => simulateProgress(achievement.id)}
                    >
                      Simulate Progress
                    </Button>
                  )}
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      </FadeIn>
    </div>
  );
};

export default AchievementSystem;