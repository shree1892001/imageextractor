import React, { useState } from 'react';
import styled from 'styled-components';

interface FormElementProps {
  id: string;
  type: string;
  content: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  properties: {
    required?: boolean;
    readonly?: boolean;
    defaultValue?: string | boolean;
    options?: string[];
    fieldType: string;
    maxLength?: number;
    multiline?: boolean;
  };
}

const FormElement: React.FC<FormElementProps> = ({ id, type, content, position, properties }) => {
  const [value, setValue] = useState(properties.defaultValue || '');

  const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setValue(event.target.value);
  };

  let element:
    | JSX.Element
    | null
    | undefined = null;
  
  switch (properties.fieldType) {
    case 'Text':
      element = (
        <Input
          id={id}
          type="text"
          value={value}
          onChange={handleChange}
          readOnly={properties.readonly}
          maxLength={properties.maxLength}
        />
      );
      break;
    case 'CheckBox':
      element = (
        <Checkbox
          id={id}
          type="checkbox"
          checked={value as boolean}
          onChange={(e) => setValue(e.target.checked)}
        />
      );
      break;
    default:
      element = null; 
  }

  return (
    <StyledFormElement style={{ ...position }}>
      {element}
    </StyledFormElement>
  );
};

export default FormElement;

const StyledFormElement = styled.div`
  position: absolute;
  /* Add more styles as needed based on PDF analysis*/
`;

const Input = styled.input`
  width: 100%;
  height: 100%;
  border: none;
  background: transparent;
`;

const Checkbox = styled.input`
  width: auto;
  height: auto;
`;
