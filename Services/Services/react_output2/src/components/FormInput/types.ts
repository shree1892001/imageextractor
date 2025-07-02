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
  x: number;
  y: number;
  width: number;
  height: number;
}