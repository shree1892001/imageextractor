import React, { useState } from 'react';
import styled from 'styled-components';
import { FormElement } from './FormElement';

interface FormPageProps {
  elements: any[];
}

const FormPage: React.FC<FormPageProps> = ({ elements }) => {
  const [formData, setFormData] = useState({});

  const handleInputChange = (id: string, value: any) => {
    setFormData({ ...formData, [id]: value });
  };

  const handleCheckboxChange = (id: string, checked: boolean) => {
    setFormData({ ...formData, [id]: checked });
  };

  return (
    <PageContainer>
      {elements.map((element) => (
        <FormElement
          key={element.id}
          element={element}
          onChange={handleInputChange}
          onCheckboxChange={handleCheckboxChange}
        />
      ))}
    </PageContainer>
  );
};

export default FormPage;