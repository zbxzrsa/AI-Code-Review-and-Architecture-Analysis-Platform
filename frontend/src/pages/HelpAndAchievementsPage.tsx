import React, { useState } from 'react';
import { Tabs, Typography, Card } from 'antd';
import { QuestionCircleOutlined, TrophyOutlined } from '@ant-design/icons';
import HelpCenter from '../components/help/HelpCenter';
import AchievementSystem from '../components/achievements/AchievementSystem';

const { Title } = Typography;


const HelpAndAchievementsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('help');
  
  // Simulated user ID
  const userId = 'current-user-id';
  
  // Handle contact support
  const handleContactSupport = () => {
    console.log('Contact support');
  };
  
  // Handle achievement completed
  const handleAchievementCompleted = (achievement: any) => {
    console.log('Achievement completed:', achievement);
  };
  
  return (
    <div className="help-achievements-page">
      <Card>
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          size="large"
          tabBarStyle={{ marginBottom: 24 }}
          items={[
            {
              key: 'help',
              label: (
                <span>
                  <QuestionCircleOutlined /> Help Center
                </span>
              ),
              children: <HelpCenter onContactSupport={handleContactSupport} />
            },
            {
              key: 'achievements',
              label: (
                <span>
                  <TrophyOutlined /> Achievements & Progress
                </span>
              ),
              children: (
                <>
                  <Title level={4}>Your Achievements & Learning Progress</Title>
                  <AchievementSystem 
                    userId={userId} 
                    onAchievementCompleted={handleAchievementCompleted} 
                  />
                </>
              )
            }
          ]}
        />
      </Card>
    </div>
  );
};

export default HelpAndAchievementsPage;