import React from 'react';
import {
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';

interface Issue {
  type: 'security' | 'quality' | 'performance' | 'style' | 'architecture';
  severity: 'low' | 'medium' | 'high' | 'critical';
  line: number;
  end_line?: number;
  message: string;
  suggestion: string;
  confidence?: number;
  code_snippet?: string;
}

interface ReviewResultsProps {
  issues: Issue[];
  score: number;
  processingTime: number;
  version: string;
  modelUsed: string;
  language: string;
  focusAreas: string[];
  cached?: boolean;
}

const ReviewResults: React.FC<ReviewResultsProps> = ({
  issues,
  score,
  processingTime,
  version,
  modelUsed,
  language,
  focusAreas,
  cached = false,
}) => {
  const getSeverityColor = (severity: Issue['severity']) => {
    switch (severity) {
      case 'critical':
        return 'text-red-700 bg-red-50 border-red-200';
      case 'high':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getTypeIcon = (type: Issue['type']) => {
    switch (type) {
      case 'security':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'quality':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case 'performance':
        return <InformationCircleIcon className="h-5 w-5 text-blue-500" />;
      case 'style':
        return <InformationCircleIcon className="h-5 w-5 text-gray-500" />;
      case 'architecture':
        return <InformationCircleIcon className="h-5 w-5 text-purple-500" />;
      default:
        return <InformationCircleIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 80) return 'text-blue-600';
    if (score >= 70) return 'text-yellow-600';
    if (score >= 60) return 'text-orange-600';
    return 'text-red-600';
  };

  const getScoreGrade = (score: number) => {
    if (score >= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    return 'F';
  };

  const groupedIssues = issues.reduce(
    (acc, issue) => {
      if (!acc[issue.type]) {
        acc[issue.type] = [];
      }
      acc[issue.type].push(issue);
      return acc;
    },
    {} as Record<string, Issue[]>
  );

  const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
  const sortedIssues = issues.sort((a, b) => {
    const severityDiff = severityOrder[a.severity] - severityOrder[b.severity];
    if (severityDiff !== 0) return severityDiff;
    return a.line - b.line;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Review Results</h2>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500">
              Version: <span className="font-medium text-gray-700">{version}</span>
            </div>
            <div className="text-sm text-gray-500">
              Model: <span className="font-medium text-gray-700">{modelUsed}</span>
            </div>
            {cached && (
              <div className="text-sm text-green-600 flex items-center">
                <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-8-8a1 1 0 000-1.414l8-8z"
                    clipRule="evenodd"
                  />
                </svg>
                Cached
              </div>
            )}
          </div>
        </div>

        {/* Score and Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className={`text-3xl font-bold ${getScoreColor(score)}`}>{score}</div>
            <div className="text-sm text-gray-500">Overall Score</div>
            <div className={`text-lg font-semibold ${getScoreColor(score)}`}>
              Grade {getScoreGrade(score)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-semibold text-gray-700">{issues.length}</div>
            <div className="text-sm text-gray-500">Issues Found</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-semibold text-gray-700">{processingTime.toFixed(2)}s</div>
            <div className="text-sm text-gray-500">Processing Time</div>
          </div>
        </div>

        {/* Focus Areas */}
        {focusAreas.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="text-sm text-gray-500 mb-2">Focus Areas:</div>
            <div className="flex flex-wrap gap-2">
              {focusAreas.map(area => (
                <span
                  key={area}
                  className="px-3 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full"
                >
                  {area}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Issues by Type */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Issues by Type</h3>
        </div>
        <div className="divide-y divide-gray-200">
          {Object.entries(groupedIssues).map(([type, typeIssues]) => (
            <div key={type} className="p-4">
              <div className="flex items-center mb-3">
                {getTypeIcon(type as Issue['type'])}
                <h4 className="ml-2 text-lg font-medium text-gray-900 capitalize">
                  {type} ({typeIssues.length})
                </h4>
              </div>

              <div className="space-y-2">
                {typeIssues
                  .sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity])
                  .slice(0, 5) // Show first 5 issues per type
                  .map((issue, index) => (
                    <div
                      key={index}
                      className={`
                        border rounded-lg p-4
                        ${getSeverityColor(issue.severity)}
                      `}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center mb-2">
                            <span className="text-xs font-medium uppercase tracking-wide opacity-75">
                              {issue.severity}
                            </span>
                            <span className="ml-2 text-sm font-medium">
                              Line {issue.line}
                              {issue.end_line &&
                                issue.end_line !== issue.line &&
                                `-${issue.end_line}`}
                            </span>
                          </div>
                          <p className="text-sm font-medium text-gray-900 mb-2">{issue.message}</p>
                          {issue.suggestion && (
                            <div className="text-sm">
                              <span className="font-medium">Suggestion:</span> {issue.suggestion}
                            </div>
                          )}
                          {issue.confidence && (
                            <div className="text-xs text-gray-500 mt-1">
                              Confidence: {(issue.confidence * 100).toFixed(1)}%
                            </div>
                          )}
                        </div>
                      </div>

                      {issue.code_snippet && (
                        <div className="mt-3 pt-3 border-t border-current border-opacity-20">
                          <div className="text-xs text-gray-500 mb-2 font-mono">Code:</div>
                          <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded overflow-x-auto font-mono">
                            <code>{issue.code_snippet}</code>
                          </pre>
                        </div>
                      )}
                    </div>
                  ))}
              </div>

              {typeIssues.length > 5 && (
                <div className="text-center mt-4 pt-4 border-t border-current border-opacity-20">
                  <button className="text-sm text-blue-600 hover:text-blue-800 font-medium">
                    View all {typeIssues.length} {type} issues →
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* All Issues List */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 mt-6">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">All Issues ({issues.length})</h3>
        </div>
        <div className="max-h-96 overflow-y-auto">
          <div className="divide-y divide-gray-200">
            {sortedIssues.map((issue, index) => (
              <div key={index} className="p-4 hover:bg-gray-50">
                <div className="flex items-start">
                  <div className="flex-shrink-0 mr-3 mt-1">{getTypeIcon(issue.type)}</div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center mb-2">
                      <span
                        className={`
                        inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                        ${getSeverityColor(issue.severity)}
                      `}
                      >
                        {issue.severity.toUpperCase()}
                      </span>
                      <span className="ml-2 text-sm text-gray-500">
                        {issue.type} • Line {issue.line}
                      </span>
                      {issue.confidence && (
                        <span className="ml-2 text-xs text-gray-400">
                          ({(issue.confidence * 100).toFixed(0)}% confidence)
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-900 mb-2">{issue.message}</p>
                    {issue.suggestion && (
                      <div className="text-sm">
                        <span className="font-medium text-gray-700">Fix:</span> {issue.suggestion}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Language Info */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Language:</span>
            <span className="ml-2 font-medium text-gray-900 capitalize">{language}</span>
          </div>
          <div>
            <span className="text-gray-500">Total Issues:</span>
            <span className="ml-2 font-medium text-gray-900">{issues.length}</span>
          </div>
          <div>
            <span className="text-gray-500">Critical Issues:</span>
            <span className="ml-2 font-medium text-red-600">
              {issues.filter(i => i.severity === 'critical').length}
            </span>
          </div>
          <div>
            <span className="text-gray-500">High Issues:</span>
            <span className="ml-2 font-medium text-red-600">
              {issues.filter(i => i.severity === 'high').length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReviewResults;
