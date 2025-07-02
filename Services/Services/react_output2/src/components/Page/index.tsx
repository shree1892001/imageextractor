import React from 'react';
import FormInput from './FormInput';
import FormCheckbox from './FormCheckbox';

interface PageProps {
  pageNumber: number;
  elements: any[];
}

const Page: React.FC<PageProps> = ({ pageNumber, elements }) => {
  return (
    <div style={{width: '612px', height: '792px', border: '1px solid black', position: 'relative'}}> 
      {elements.map((element) => {
        if (element.type === 'form') {
          if(element.properties.fieldType === 'Text'){
            return <FormInput key={element.id} {...element.position} {...element.properties} id={element.id} type="text" />;
          } else if (element.properties.fieldType === 'CheckBox') {
            return <FormCheckbox key={element.id} {...element.position} {...element.properties} id={element.id} />;
          } else {
            return null; // Handle other form field types
          }
        } else if (element.type === 'text') {
          return (
            <div
              key={element.id}
              style={{
                position: 'absolute',
                left: element.position.x,
                top: element.position.y,
                fontSize: element.properties.fontSize,
                color: element.properties.color,
                fontFamily: element.properties.font
              }}
            >
              {element.content}
            </div>
          );
        }
        return null; // Handle other element types
      })}
    </div>
  );
};

export default Page;