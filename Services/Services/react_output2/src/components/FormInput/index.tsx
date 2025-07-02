import React, { useState } from 'react';
import styled from 'styled-components';

interface FormInputProps {
  id: string;
  type: string;
  label?: string;
  placeholder?: string;
  required?: boolean;
  readonly?: boolean;
  defaultValue?: string;
  maxLength?: number;
  multiline?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void;
}

const StyledInput = styled.input`
  width: ${({ width }) => width}px;
  height: ${({ height }) => height}px;
  border: 1px solid #ccc;
  padding: 5px;
  margin-bottom: 10px;
  box-sizing: border-box;
`;

const StyledTextarea = styled.textarea`
  width: ${({ width }) => width}px;
  height: ${({ height }) => height}px;
  border: 1px solid #ccc;
  padding: 5px;
  margin-bottom: 10px;
  box-sizing: border-box;
`;

const FormInput: React.FC<FormInputProps> = ({ id, type, label, placeholder, required, readonly, defaultValue, maxLength, multiline, onChange, ...props }) => {
  const [value, setValue] = useState(defaultValue || '');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setValue(e.target.value);
    if (onChange) onChange(e);
  };

  return (
    <div style={{ position: 'absolute', left: props.x, top: props.y }}>
      {label && <label htmlFor={id}>{label}</label>}
      {type === 'text' ? (
        <StyledInput
          id={id}
          type={type}
          placeholder={placeholder}
          required={required}
          readOnly={readonly}
          value={value}
          maxLength={maxLength}
          onChange={handleChange}
          width={props.width}
          height={props.height}
        />
      ) : type === 'textarea' ? (
        <StyledTextarea
          id={id}
          placeholder={placeholder}
          required={required}
          readOnly={readonly}
          value={value}
          onChange={handleChange}
          width={props.width}
          height={props.height}
        />
      ) : null}
    </div>
  );
};

export default FormInput;