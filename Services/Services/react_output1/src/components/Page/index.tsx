import React from 'react';
import FormElement from './FormElement';
import TextElement from './TextElement';

interface PageProps {
  pageNumber: number;
  elements: (FormElementProps | TextElementProps)[];
}

const Page: React.FC<PageProps> = ({ pageNumber, elements }) => {
  return (
    <div>
      {elements.map((element) => {
        if (element.type === 'form') {
          return <FormElement key={element.id} {...element} />;
        } else if (element.type === 'text') {
          return <TextElement key={element.id} {...element} />;
        }
        return null; // Handle other element types if needed
      })}
    </div>
  );
};

export default Page;