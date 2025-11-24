// Metrics service for dashboard data
export interface Metrics {
  kpis: {
    analyzedProjects: number;
    issuesFound: number;
    vulnerabilities: number;
    coverage: number;
  };
  trends: {
    issueTrend: number[];
    analysisTime: number[];
  };
}

export const fetchMetrics = async (timeRange: number): Promise<Metrics> => {
  // Simulate API call
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        kpis: {
          analyzedProjects: 42,
          issuesFound: 156,
          vulnerabilities: 8,
          coverage: 78,
        },
        trends: {
          issueTrend: [12, 19, 15, 25, 22, 30, 28],
          analysisTime: [2.5, 3.1, 2.8, 3.5, 3.2, 2.9, 3.0],
        },
      });
    }, 1000);
  });
};

export default {
  fetchMetrics,
};
