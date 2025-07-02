import React from 'react';
import styled from 'styled-components';

const Page4Container = styled.div`
  width: 612px;
  height: 792px;
  position: relative;
`;

const TextElement = styled.div<{ x: number; y: number; width: number; height: number; font?: string; fontSize?: number; color?: string; }>`
  position: absolute;
  left: ${(props) => props.x}px;
  top: ${(props) => props.y}px;
  width: ${(props) => props.width}px;
  height: ${(props) => props.height}px;
  font-family: ${(props) => props.font || 'inherit'};
  font-size: ${(props) => props.fontSize}px;
  color: ${(props) => props.color};
`;

const Page4: React.FC = () => {
  return (
    <Page4Container>
      <TextElement x={2.8350000381469727} y={789.7000122070312} width={16.17000102996826} height={1.3740234375} font="Helvetica" fontSize={1.0} color="#000000">
        Powered by TCPDF (www.tcpdf.org)
      </TextElement>
    </Page4Container>
  );
};

export default Page4;