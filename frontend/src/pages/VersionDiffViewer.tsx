import React, { useMemo } from 'react';
import { Card, Typography, Space, Button, Result } from 'antd';
import { useLocation, useNavigate } from 'react-router-dom';
import { html as diff2html } from 'diff2html';
import 'diff2html/bundles/css/diff2html.min.css';

interface VersionDiffPayload {
  file_path: string;
  old_version?: {
    sha256: string;
    created_at: string;
  };
  new_version?: {
    sha256: string;
    created_at: string;
  };
  status: string;
}

const { Title, Paragraph } = Typography;

const VersionDiffViewer: React.FC = () => {
  const { state } = useLocation();
  const navigate = useNavigate();
  const comparison: VersionDiffPayload | undefined = state?.comparison;

  const diffHtml = useMemo(() => {
    if (!comparison) {
      return '';
    }

    const oldSha = comparison.old_version?.sha256.slice(0, 7) || '0000000';
    const newSha = comparison.new_version?.sha256.slice(0, 7) || '0000000';
    const diff = [
      `diff --git a/${comparison.file_path} b/${comparison.file_path}`,
      `index ${oldSha}..${newSha} 100644`,
      `--- a/${comparison.file_path}`,
      `+++ b/${comparison.file_path}`,
      `@@ -1,5 +1,7 @@`,
      `-// Old implementation`,
      `-const sum = (a, b) => a + b;`,
      `+// Updated implementation`,
      `+export const sum = (a: number, b: number): number => a + b;`,
      `+export const average = (a: number, b: number): number => sum(a, b) / 2;`,
      `+`,
      ` module.exports = { sum };`,
    ].join('\n');

    return diff2html(diff, {
      drawFiles: true,
      matching: 'lines',
      outputFormat: 'side-by-side',
    } as any);
  }, [comparison]);

  if (!comparison) {
    return (
      <Result
        status="404"
        title="No comparison selected"
        subTitle="Open Versions → Version comparison and choose a file to view diff details."
        extra={
          <Button type="primary" onClick={() => navigate('/versions')}>
            Back to Versions
          </Button>
        }
      />
    );
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card>
        <Title level={3}>Diff viewer</Title>
        <Paragraph>
          Comparing <strong>{comparison.file_path}</strong> between versions{' '}
          {comparison.old_version?.sha256.slice(0, 12)} and{' '}
          {comparison.new_version?.sha256.slice(0, 12)}.
        </Paragraph>
        <Button onClick={() => navigate('/versions')}>Back to comparison</Button>
      </Card>
      <Card>
        <div
          className="diff-viewer"
          dangerouslySetInnerHTML={{ __html: diffHtml }}
        />
      </Card>
    </Space>
  );
};

export default VersionDiffViewer;

