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