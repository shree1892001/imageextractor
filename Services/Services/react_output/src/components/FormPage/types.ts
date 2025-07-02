interface FormElementProps {
  element: any;
  onChange: (id: string, value: any) => void;
  onCheckboxChange: (id: string, checked: boolean) => void;
}

interface FormPageProps {
  elements: any[];
}