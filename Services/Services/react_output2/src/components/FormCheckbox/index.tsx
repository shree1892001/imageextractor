import React, { useState } from 'react';
import styled from 'styled-components';

interface FormCheckboxProps {
  id: string;
  label?: string;
  defaultValue?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  x: number;
  y: number;
  width: number;
  height: number;
}

const StyledCheckbox = styled.input`
  margin-right: 5px;
`;

const FormCheckbox: React.FC<FormCheckboxProps> = ({ id, label, defaultValue, onChange, x, y, width, height }) => {
  const [checked, setChecked] = useState(defaultValue || false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setChecked(e.target.checked);
    if (onChange) onChange(e);
  };

  return (
    <label style={{ position: 'absolute', left: x, top: y }} >
      <StyledCheckbox type="checkbox" id={id} checked={checked} onChange={handleChange} />
      {label}
    </label>
  );
};

export default FormCheckbox;