import React, { useState } from 'react';
import styled from 'styled-components';

const FormContainer = styled.div`
  position: relative;
`;

const FormElement = styled.div`
  position: absolute;
  left: ${({ x }) => `${x}px`};
  top: ${({ y }) => `${y}px`};
  width: ${({ width }) => `${width}px`};
  height: ${({ height }) => `${height}px`};
`;

const Input = styled.input`
  width: 100%;
  height: 100%;
  border: 1px solid #ccc;
  padding: 0.5rem;
  box-sizing: border-box;
`;

const TextArea = styled.textarea`
  width: 100%;
  height: 100%;
  border: 1px solid #ccc;
  padding: 0.5rem;
  box-sizing: border-box;
  resize: ${({ multiline }) => (multiline ? 'vertical' : 'none')};
`;

const Checkbox = styled.input`
  width: auto;
  height: auto;  
`;

interface FormElementProps {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  fieldType: string;
  multiline?: boolean;
  required?: boolean;
  defaultValue?: string;
  options?: string[];
}

const FormElementComponent: React.FC<FormElementProps> = ({ id, x, y, width, height, fieldType, multiline, required, defaultValue, options }) => {
  const [value, setValue] = useState(defaultValue || '');

  const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setValue(event.target.value);
  };

  let inputElement;
  switch (fieldType) {
    case 'Text':
      inputElement = <Input type="text" value={value} onChange={handleChange} required={required} />;
      break;
    case 'CheckBox':
      inputElement = <Checkbox type="checkbox" checked={value === 'true'} onChange={handleChange} />;
      break;
    default:
      inputElement = null;
  }

  return (
    <FormElement x={x} y={y} width={width} height={height}>
      {inputElement}
    </FormElement>
  );
};

const FormPage1: React.FC = () => {
  const formElementsData = [
    // ... (Your form element data here)
  ];
  return (
    <FormContainer>
      {formElementsData.map((element) => (
        <FormElementComponent key={element.id} {...element} />
      ))}
    </FormContainer>
  );
};

export default FormPage1;
