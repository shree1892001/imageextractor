import React from 'react';
import styled from 'styled-components';

interface FormElementProps {
  element: any;
  onChange: (id: string, value: any) => void;
  onCheckboxChange: (id: string, checked: boolean) => void;
}

const FormElement: React.FC<FormElementProps> = ({ element, onChange, onCheckboxChange }) => {
  const { type, position, properties, content } = element;

  const handleValueChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(element.id, e.target.value);
  };

  const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onCheckboxChange(element.id, e.target.checked);
  };

  switch (type) {
    case 'form':
      switch (properties.fieldType) {
        case 'Text':
          return (
            <InputContainer style={position}>
              <input
                type="text"
                id={content}
                value={element.value || ''}
                onChange={handleValueChange}
                style={{ width: properties.width, height: properties.height }}
              />
            </InputContainer>
          );
        case 'CheckBox':
          return (
            <CheckboxContainer style={position}>
              <input
                type="checkbox"
                id={content}
                checked={element.value || false}
                onChange={handleCheckboxChange}
              />
            </CheckboxContainer>
          );
        default:
          return null;
      }
    case 'text':
      return <TextContainer style={position}>{element.content}</TextContainer>;
    default:
      return null;
  }
};

export default FormElement;