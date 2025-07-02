interface FormElementProps {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  fieldType: string;  // 'Text', 'CheckBox'
  multiline?: boolean;
  required?: boolean;
  defaultValue?: string;
  options?: string[];
}

// Add other necessary types as needed
