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