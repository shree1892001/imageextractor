import React from 'react';
import styled from 'styled-components';
import { Form, Input, Checkbox } from 'antd';

const Page1Container = styled.div`
  width: 612px;
  height: 792px;
  position: relative;
`;

const FormElement = styled.div<{ x: number; y: number; width: number; height: number }>`
  position: absolute;
  left: ${({ x }) => `${x}px`};
  top: ${({ y }) => `${y}px`};
  width: ${({ width }) => `${width}px`};
  height: ${({ height }) => `${height}px`};
`;

const TextElement = styled.div<{ x: number; y: number; width: number; height: number; font?: string; fontSize?: number; color?: string }>`
  position: absolute;
  left: ${({ x }) => `${x}px`};
  top: ${({ y }) => `${y}px`};
  width: ${({ width }) => `${width}px`};
  height: ${({ height }) => `${height}px`};
  font-family: ${({ font }) => font || 'Helvetica'};
  font-size: ${({ fontSize }) => fontSize}px;
  color: ${({ color }) => color};
`;

interface FormElementProps {
  id: string;
  fieldType: string;
  required?: boolean;
  readonly?: boolean;
  defaultValue?: string;
  options?: string[];
  maxLength?: number;
  multiline?: boolean;
}

const FormComponent: React.FC<FormElementProps> = ({ id, fieldType, required, readonly, defaultValue, options, maxLength, multiline }) => {
  switch (fieldType) {
    case 'Text':
      return (
        <Form.Item name={id} rules={[{ required }]} initialValue={defaultValue}>
          <Input readOnly={readonly} maxLength={maxLength} style={{ width: '100%' }} />
        </Form.Item>
      );
    case 'CheckBox':
      return (
        <Form.Item name={id} valuePropName="checked">
          <Checkbox checked={defaultValue === 'true'} disabled={readonly}>
            {/* Checkbox content here, can be customized based on analysis */}
          </Checkbox>
        </Form.Item>
      );
    default:
      return null;
  }
};

const Page1: React.FC = () => {
  return (
    <Page1Container>
      <Form>
        {data.pages[0].elements.map((element) =>
          element.type === 'form' ? (
            <FormElement key={element.id} x={element.position.x} y={element.position.y} width={element.position.width} height={element.position.height}>
              <FormComponent {...element.properties} id={element.id} />
            </FormElement>
          ) : element.type === 'text' ? (
            <TextElement key={element.id} x={element.position.x} y={element.position.y} width={element.position.width} height={element.position.height} {...element.properties}>
              {element.content}
            </TextElement>
          ) : null
        )}
      </Form>
    </Page1Container>
  );
};

export default Page1;
