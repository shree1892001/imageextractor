import React from 'react';
import styled from 'styled-components';
import { Form, Input, Checkbox } from 'antd';

const Page1Container = styled.div`
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

const TextElement = styled.div<{ x: number; y: number; width: number; height: number; font?: string; fontSize?: number; color?: string; }>`
  position: absolute;
  left: ${(props) => props.x}px;
  top: ${(props) => props.y}px;
  width: ${(props) => props.width}px;
  height: ${(props) => props.height}px;
  font-family: ${(props) => props.font || 'inherit'};
  font-size: ${(props) => props.fontSize}px;
  color: ${(props) => props.color};
`;

const Page1: React.FC = () => {
  const onFinish = (values: any) => {
    console.log('Success:', values);
  };

  const onFinishFailed = (errorInfo: any) => {
    console.log('Failed:', errorInfo);
  };

  return (
    <Page1Container>
      <Form name="basic" initialValues={{ remember: true }}
            onFinish={onFinish}
            onFinishFailed={onFinishFailed}>
        <FormElement x={41.032257080078125} y={113.03228759765625} width={178.0575714111328} height={11.22576904296875}>
          <Form.Item name="f37a322f-328a-45cb-854f-99d1788c9666">
            <Input />
          </Form.Item>
        </FormElement>
        {/* ... other form elements ... */}
        <FormElement x={41.032257080078125} y={166.06451416015625} width={10.83871078491211} height={10.83868408203125}>
          <Form.Item name="971d0914-7110-4749-aa16-67e7f7b263a7">
            <Checkbox /> 
          </Form.Item>
        </FormElement>
        <TextElement x={43.40761184692383} y={166.9647674560547} width={6.096000671386719} height={7.9999847412109375} font="ZapfDingbats" fontSize={7.999990940093994} color="#000000">
          J
        </TextElement>
        {/* ... other form and text elements ... */}
      </Form>
    </Page1Container>
  );
};

export default Page1;