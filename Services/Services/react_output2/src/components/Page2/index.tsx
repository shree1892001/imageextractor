import React from 'react';
import styled from 'styled-components';
import { Form, Input } from 'antd';

const Page2Container = styled.div`
  width: 612px;
  height: 792px;
  position: relative;
`;

const FormElement = styled.div<{ x: number; y: number; width: number; height: number; }>`
  position: absolute;
  left: ${(props) => props.x}px;
  top: ${(props) => props.y}px;
  width: ${(props) => props.width}px;
  height: ${(props) => props.height}px;
`;

const Page2: React.FC = () => {
  const onFinish = (values: any) => {
    console.log('Success:', values);
  };

  const onFinishFailed = (errorInfo: any) => {
    console.log('Failed:', errorInfo);
  };

  return (
    <Page2Container>
      <Form name="basic" initialValues={{ remember: true }}
            onFinish={onFinish}
            onFinishFailed={onFinishFailed}>
        <FormElement x={57.290321350097656} y={573.35498046875} width={54.967742919921875} height={11.1611328125}>
          <Form.Item name="b51880d7-5a33-4ce0-9ac3-16b82e1881b6">
            <Input />
          </Form.Item>
        </FormElement>
        {/* ... other form elements ... */}
      </Form>
    </Page2Container>
  );
};

export default Page2;